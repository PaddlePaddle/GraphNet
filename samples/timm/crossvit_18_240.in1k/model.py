import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_x_: torch.Tensor,
        L_self_modules_patch_embed_modules_0_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed_modules_0_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_parameters_cls_token_0_: torch.nn.parameter.Parameter,
        L_self_parameters_pos_embed_0_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed_modules_1_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed_modules_1_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_parameters_cls_token_1_: torch.nn.parameter.Parameter,
        L_self_parameters_pos_embed_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_projs_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_projs_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_projs_modules_0_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_projs_modules_0_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_projs_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_projs_modules_1_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_projs_modules_1_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_projs_modules_1_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wq_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wq_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wk_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wk_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wq_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wq_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wk_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wk_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_projs_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_projs_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_projs_modules_0_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_projs_modules_0_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_projs_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_projs_modules_1_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_projs_modules_1_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_projs_modules_1_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wq_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wq_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wk_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wk_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wq_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wq_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wk_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wk_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_projs_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_projs_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_projs_modules_0_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_projs_modules_0_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_projs_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_projs_modules_1_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_projs_modules_1_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_projs_modules_1_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wq_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wq_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wk_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wk_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wq_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wq_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wk_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wk_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_norm_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_norm_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_norm_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_norm_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_x_ = L_x_
        l_self_modules_patch_embed_modules_0_modules_proj_parameters_weight_ = (
            L_self_modules_patch_embed_modules_0_modules_proj_parameters_weight_
        )
        l_self_modules_patch_embed_modules_0_modules_proj_parameters_bias_ = (
            L_self_modules_patch_embed_modules_0_modules_proj_parameters_bias_
        )
        l_self_parameters_cls_token_0_ = L_self_parameters_cls_token_0_
        l_self_parameters_pos_embed_0_ = L_self_parameters_pos_embed_0_
        l_self_modules_patch_embed_modules_1_modules_proj_parameters_weight_ = (
            L_self_modules_patch_embed_modules_1_modules_proj_parameters_weight_
        )
        l_self_modules_patch_embed_modules_1_modules_proj_parameters_bias_ = (
            L_self_modules_patch_embed_modules_1_modules_proj_parameters_bias_
        )
        l_self_parameters_cls_token_1_ = L_self_parameters_cls_token_1_
        l_self_parameters_pos_embed_1_ = L_self_parameters_pos_embed_1_
        l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_norm1_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_norm1_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_norm1_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_norm1_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_norm2_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_norm2_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_norm2_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_norm2_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_norm1_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_norm1_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_norm1_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_norm1_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_norm2_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_norm2_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_norm2_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_norm2_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_norm1_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_norm1_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_norm1_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_norm1_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_norm2_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_norm2_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_norm2_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_norm2_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_norm1_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_norm1_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_norm1_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_norm1_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_norm2_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_norm2_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_norm2_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_norm2_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_norm1_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_norm1_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_norm1_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_norm1_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_norm2_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_norm2_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_norm2_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_norm2_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_norm1_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_norm1_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_norm1_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_norm1_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_attn_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_attn_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_norm2_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_norm2_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_norm2_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_norm2_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_0_modules_projs_modules_0_modules_0_parameters_weight_ = L_self_modules_blocks_modules_0_modules_projs_modules_0_modules_0_parameters_weight_
        l_self_modules_blocks_modules_0_modules_projs_modules_0_modules_0_parameters_bias_ = L_self_modules_blocks_modules_0_modules_projs_modules_0_modules_0_parameters_bias_
        l_self_modules_blocks_modules_0_modules_projs_modules_0_modules_2_parameters_weight_ = L_self_modules_blocks_modules_0_modules_projs_modules_0_modules_2_parameters_weight_
        l_self_modules_blocks_modules_0_modules_projs_modules_0_modules_2_parameters_bias_ = L_self_modules_blocks_modules_0_modules_projs_modules_0_modules_2_parameters_bias_
        l_self_modules_blocks_modules_0_modules_projs_modules_1_modules_0_parameters_weight_ = L_self_modules_blocks_modules_0_modules_projs_modules_1_modules_0_parameters_weight_
        l_self_modules_blocks_modules_0_modules_projs_modules_1_modules_0_parameters_bias_ = L_self_modules_blocks_modules_0_modules_projs_modules_1_modules_0_parameters_bias_
        l_self_modules_blocks_modules_0_modules_projs_modules_1_modules_2_parameters_weight_ = L_self_modules_blocks_modules_0_modules_projs_modules_1_modules_2_parameters_weight_
        l_self_modules_blocks_modules_0_modules_projs_modules_1_modules_2_parameters_bias_ = L_self_modules_blocks_modules_0_modules_projs_modules_1_modules_2_parameters_bias_
        l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_norm1_parameters_weight_ = L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_norm1_parameters_weight_
        l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_norm1_parameters_bias_ = L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_norm1_parameters_bias_
        l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wq_parameters_weight_ = L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wq_parameters_weight_
        l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wq_parameters_bias_ = L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wq_parameters_bias_
        l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wk_parameters_weight_ = L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wk_parameters_weight_
        l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wk_parameters_bias_ = L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wk_parameters_bias_
        l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wv_parameters_weight_ = L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wv_parameters_weight_
        l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wv_parameters_bias_ = L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wv_parameters_bias_
        l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_0_parameters_weight_ = L_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_0_parameters_weight_
        l_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_0_parameters_bias_ = L_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_0_parameters_bias_
        l_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_2_parameters_weight_ = L_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_2_parameters_weight_
        l_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_2_parameters_bias_ = L_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_2_parameters_bias_
        l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_norm1_parameters_weight_ = L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_norm1_parameters_weight_
        l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_norm1_parameters_bias_ = L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_norm1_parameters_bias_
        l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wq_parameters_weight_ = L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wq_parameters_weight_
        l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wq_parameters_bias_ = L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wq_parameters_bias_
        l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wk_parameters_weight_ = L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wk_parameters_weight_
        l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wk_parameters_bias_ = L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wk_parameters_bias_
        l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wv_parameters_weight_ = L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wv_parameters_weight_
        l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wv_parameters_bias_ = L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wv_parameters_bias_
        l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_0_parameters_weight_ = L_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_0_parameters_weight_
        l_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_0_parameters_bias_ = L_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_0_parameters_bias_
        l_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_2_parameters_weight_ = L_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_2_parameters_weight_
        l_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_2_parameters_bias_ = L_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_2_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm1_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm1_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm1_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm1_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm2_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm2_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm2_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm2_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm1_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm1_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm1_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm1_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm2_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm2_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm2_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm2_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm1_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm1_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm1_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm1_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm2_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm2_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm2_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm2_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_norm1_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_norm1_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_norm1_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_norm1_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_norm2_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_norm2_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_norm2_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_norm2_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_norm1_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_norm1_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_norm1_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_norm1_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_norm2_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_norm2_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_norm2_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_norm2_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_norm1_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_norm1_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_norm1_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_norm1_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_attn_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_attn_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_norm2_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_norm2_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_norm2_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_norm2_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_1_modules_projs_modules_0_modules_0_parameters_weight_ = L_self_modules_blocks_modules_1_modules_projs_modules_0_modules_0_parameters_weight_
        l_self_modules_blocks_modules_1_modules_projs_modules_0_modules_0_parameters_bias_ = L_self_modules_blocks_modules_1_modules_projs_modules_0_modules_0_parameters_bias_
        l_self_modules_blocks_modules_1_modules_projs_modules_0_modules_2_parameters_weight_ = L_self_modules_blocks_modules_1_modules_projs_modules_0_modules_2_parameters_weight_
        l_self_modules_blocks_modules_1_modules_projs_modules_0_modules_2_parameters_bias_ = L_self_modules_blocks_modules_1_modules_projs_modules_0_modules_2_parameters_bias_
        l_self_modules_blocks_modules_1_modules_projs_modules_1_modules_0_parameters_weight_ = L_self_modules_blocks_modules_1_modules_projs_modules_1_modules_0_parameters_weight_
        l_self_modules_blocks_modules_1_modules_projs_modules_1_modules_0_parameters_bias_ = L_self_modules_blocks_modules_1_modules_projs_modules_1_modules_0_parameters_bias_
        l_self_modules_blocks_modules_1_modules_projs_modules_1_modules_2_parameters_weight_ = L_self_modules_blocks_modules_1_modules_projs_modules_1_modules_2_parameters_weight_
        l_self_modules_blocks_modules_1_modules_projs_modules_1_modules_2_parameters_bias_ = L_self_modules_blocks_modules_1_modules_projs_modules_1_modules_2_parameters_bias_
        l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_norm1_parameters_weight_ = L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_norm1_parameters_weight_
        l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_norm1_parameters_bias_ = L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_norm1_parameters_bias_
        l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wq_parameters_weight_ = L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wq_parameters_weight_
        l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wq_parameters_bias_ = L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wq_parameters_bias_
        l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wk_parameters_weight_ = L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wk_parameters_weight_
        l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wk_parameters_bias_ = L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wk_parameters_bias_
        l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wv_parameters_weight_ = L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wv_parameters_weight_
        l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wv_parameters_bias_ = L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wv_parameters_bias_
        l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_0_parameters_weight_ = L_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_0_parameters_weight_
        l_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_0_parameters_bias_ = L_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_0_parameters_bias_
        l_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_2_parameters_weight_ = L_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_2_parameters_weight_
        l_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_2_parameters_bias_ = L_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_2_parameters_bias_
        l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_norm1_parameters_weight_ = L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_norm1_parameters_weight_
        l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_norm1_parameters_bias_ = L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_norm1_parameters_bias_
        l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wq_parameters_weight_ = L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wq_parameters_weight_
        l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wq_parameters_bias_ = L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wq_parameters_bias_
        l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wk_parameters_weight_ = L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wk_parameters_weight_
        l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wk_parameters_bias_ = L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wk_parameters_bias_
        l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wv_parameters_weight_ = L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wv_parameters_weight_
        l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wv_parameters_bias_ = L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wv_parameters_bias_
        l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_0_parameters_weight_ = L_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_0_parameters_weight_
        l_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_0_parameters_bias_ = L_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_0_parameters_bias_
        l_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_2_parameters_weight_ = L_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_2_parameters_weight_
        l_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_2_parameters_bias_ = L_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_2_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm1_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm1_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm1_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm1_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm2_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm2_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm2_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm2_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm1_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm1_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm1_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm1_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm2_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm2_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm2_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm2_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm1_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm1_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm1_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm1_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm2_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm2_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm2_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm2_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_norm1_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_norm1_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_norm1_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_norm1_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_norm2_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_norm2_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_norm2_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_norm2_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_norm1_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_norm1_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_norm1_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_norm1_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_norm2_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_norm2_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_norm2_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_norm2_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_norm1_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_norm1_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_norm1_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_norm1_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_attn_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_attn_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_norm2_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_norm2_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_norm2_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_norm2_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_2_modules_projs_modules_0_modules_0_parameters_weight_ = L_self_modules_blocks_modules_2_modules_projs_modules_0_modules_0_parameters_weight_
        l_self_modules_blocks_modules_2_modules_projs_modules_0_modules_0_parameters_bias_ = L_self_modules_blocks_modules_2_modules_projs_modules_0_modules_0_parameters_bias_
        l_self_modules_blocks_modules_2_modules_projs_modules_0_modules_2_parameters_weight_ = L_self_modules_blocks_modules_2_modules_projs_modules_0_modules_2_parameters_weight_
        l_self_modules_blocks_modules_2_modules_projs_modules_0_modules_2_parameters_bias_ = L_self_modules_blocks_modules_2_modules_projs_modules_0_modules_2_parameters_bias_
        l_self_modules_blocks_modules_2_modules_projs_modules_1_modules_0_parameters_weight_ = L_self_modules_blocks_modules_2_modules_projs_modules_1_modules_0_parameters_weight_
        l_self_modules_blocks_modules_2_modules_projs_modules_1_modules_0_parameters_bias_ = L_self_modules_blocks_modules_2_modules_projs_modules_1_modules_0_parameters_bias_
        l_self_modules_blocks_modules_2_modules_projs_modules_1_modules_2_parameters_weight_ = L_self_modules_blocks_modules_2_modules_projs_modules_1_modules_2_parameters_weight_
        l_self_modules_blocks_modules_2_modules_projs_modules_1_modules_2_parameters_bias_ = L_self_modules_blocks_modules_2_modules_projs_modules_1_modules_2_parameters_bias_
        l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_norm1_parameters_weight_ = L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_norm1_parameters_weight_
        l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_norm1_parameters_bias_ = L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_norm1_parameters_bias_
        l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wq_parameters_weight_ = L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wq_parameters_weight_
        l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wq_parameters_bias_ = L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wq_parameters_bias_
        l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wk_parameters_weight_ = L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wk_parameters_weight_
        l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wk_parameters_bias_ = L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wk_parameters_bias_
        l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wv_parameters_bias_ = L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wv_parameters_bias_
        l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_0_parameters_weight_ = L_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_0_parameters_weight_
        l_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_0_parameters_bias_ = L_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_0_parameters_bias_
        l_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_2_parameters_weight_ = L_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_2_parameters_weight_
        l_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_2_parameters_bias_ = L_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_2_parameters_bias_
        l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_norm1_parameters_weight_ = L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_norm1_parameters_weight_
        l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_norm1_parameters_bias_ = L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_norm1_parameters_bias_
        l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wq_parameters_weight_ = L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wq_parameters_weight_
        l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wq_parameters_bias_ = L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wq_parameters_bias_
        l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wk_parameters_weight_ = L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wk_parameters_weight_
        l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wk_parameters_bias_ = L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wk_parameters_bias_
        l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wv_parameters_bias_ = L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wv_parameters_bias_
        l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_0_parameters_weight_ = L_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_0_parameters_weight_
        l_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_0_parameters_bias_ = L_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_0_parameters_bias_
        l_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_2_parameters_weight_ = L_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_2_parameters_weight_
        l_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_2_parameters_bias_ = L_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_2_parameters_bias_
        l_self_modules_norm_modules_0_parameters_weight_ = (
            L_self_modules_norm_modules_0_parameters_weight_
        )
        l_self_modules_norm_modules_0_parameters_bias_ = (
            L_self_modules_norm_modules_0_parameters_bias_
        )
        l_self_modules_norm_modules_1_parameters_weight_ = (
            L_self_modules_norm_modules_1_parameters_weight_
        )
        l_self_modules_norm_modules_1_parameters_bias_ = (
            L_self_modules_norm_modules_1_parameters_bias_
        )
        l_self_modules_head_modules_0_parameters_weight_ = (
            L_self_modules_head_modules_0_parameters_weight_
        )
        l_self_modules_head_modules_0_parameters_bias_ = (
            L_self_modules_head_modules_0_parameters_bias_
        )
        l_self_modules_head_modules_1_parameters_weight_ = (
            L_self_modules_head_modules_1_parameters_weight_
        )
        l_self_modules_head_modules_1_parameters_bias_ = (
            L_self_modules_head_modules_1_parameters_bias_
        )
        x = torch.nn.functional.interpolate(
            l_x_, size=(240, 240), mode="bicubic", align_corners=False
        )
        conv2d = torch.conv2d(
            x,
            l_self_modules_patch_embed_modules_0_modules_proj_parameters_weight_,
            l_self_modules_patch_embed_modules_0_modules_proj_parameters_bias_,
            (12, 12),
            (0, 0),
            (1, 1),
            1,
        )
        x = (
            l_self_modules_patch_embed_modules_0_modules_proj_parameters_weight_
        ) = l_self_modules_patch_embed_modules_0_modules_proj_parameters_bias_ = None
        flatten = conv2d.flatten(2)
        conv2d = None
        x_1 = flatten.transpose(1, 2)
        flatten = None
        cls_tokens = l_self_parameters_cls_token_0_.expand(1, -1, -1)
        l_self_parameters_cls_token_0_ = None
        x_ = torch.cat((cls_tokens, x_1), dim=1)
        cls_tokens = x_1 = None
        x__1 = x_ + l_self_parameters_pos_embed_0_
        x_ = l_self_parameters_pos_embed_0_ = None
        x__2 = torch.nn.functional.dropout(x__1, 0.0, False, False)
        x__1 = None
        conv2d_1 = torch.conv2d(
            l_x_,
            l_self_modules_patch_embed_modules_1_modules_proj_parameters_weight_,
            l_self_modules_patch_embed_modules_1_modules_proj_parameters_bias_,
            (16, 16),
            (0, 0),
            (1, 1),
            1,
        )
        l_x_ = (
            l_self_modules_patch_embed_modules_1_modules_proj_parameters_weight_
        ) = l_self_modules_patch_embed_modules_1_modules_proj_parameters_bias_ = None
        flatten_1 = conv2d_1.flatten(2)
        conv2d_1 = None
        x_2 = flatten_1.transpose(1, 2)
        flatten_1 = None
        cls_tokens_1 = l_self_parameters_cls_token_1_.expand(1, -1, -1)
        l_self_parameters_cls_token_1_ = None
        x__3 = torch.cat((cls_tokens_1, x_2), dim=1)
        cls_tokens_1 = x_2 = None
        x__4 = x__3 + l_self_parameters_pos_embed_1_
        x__3 = l_self_parameters_pos_embed_1_ = None
        x__5 = torch.nn.functional.dropout(x__4, 0.0, False, False)
        x__4 = None
        layer_norm = torch.nn.functional.layer_norm(
            x__2,
            (224,),
            l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_ = (None)
        linear = torch._C._nn.linear(
            layer_norm,
            l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm = l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape = linear.reshape(1, 401, 3, 7, 32)
        linear = None
        qkv = reshape.permute(2, 0, 3, 1, 4)
        reshape = None
        unbind = qkv.unbind(0)
        qkv = None
        q = unbind[0]
        k = unbind[1]
        v = unbind[2]
        unbind = None
        x_3 = torch._C._nn.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0
        )
        q = k = v = None
        transpose_2 = x_3.transpose(1, 2)
        x_3 = None
        x_4 = transpose_2.reshape(1, 401, 224)
        transpose_2 = None
        x_5 = torch._C._nn.linear(
            x_4,
            l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_4 = l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_6 = torch.nn.functional.dropout(x_5, 0.0, False, False)
        x_5 = None
        x_7 = x__2 + x_6
        x__2 = x_6 = None
        layer_norm_1 = torch.nn.functional.layer_norm(
            x_7,
            (224,),
            l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_ = (None)
        x_8 = torch._C._nn.linear(
            layer_norm_1,
            l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_1 = l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_9 = torch._C._nn.gelu(x_8, approximate="none")
        x_8 = None
        x_10 = torch.nn.functional.dropout(x_9, 0.0, False, False)
        x_9 = None
        x_11 = torch._C._nn.linear(
            x_10,
            l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_10 = l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_12 = torch.nn.functional.dropout(x_11, 0.0, False, False)
        x_11 = None
        x_13 = x_7 + x_12
        x_7 = x_12 = None
        layer_norm_2 = torch.nn.functional.layer_norm(
            x__5,
            (448,),
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_norm1_parameters_bias_ = (None)
        linear_4 = torch._C._nn.linear(
            layer_norm_2,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_2 = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_2 = linear_4.reshape(1, 197, 3, 7, 64)
        linear_4 = None
        qkv_1 = reshape_2.permute(2, 0, 3, 1, 4)
        reshape_2 = None
        unbind_1 = qkv_1.unbind(0)
        qkv_1 = None
        q_1 = unbind_1[0]
        k_1 = unbind_1[1]
        v_1 = unbind_1[2]
        unbind_1 = None
        x_14 = torch._C._nn.scaled_dot_product_attention(
            q_1, k_1, v_1, attn_mask=None, dropout_p=0.0
        )
        q_1 = k_1 = v_1 = None
        transpose_3 = x_14.transpose(1, 2)
        x_14 = None
        x_15 = transpose_3.reshape(1, 197, 448)
        transpose_3 = None
        x_16 = torch._C._nn.linear(
            x_15,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_15 = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_17 = torch.nn.functional.dropout(x_16, 0.0, False, False)
        x_16 = None
        x_18 = x__5 + x_17
        x__5 = x_17 = None
        layer_norm_3 = torch.nn.functional.layer_norm(
            x_18,
            (448,),
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_norm2_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_norm2_parameters_bias_ = (None)
        x_19 = torch._C._nn.linear(
            layer_norm_3,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_3 = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_20 = torch._C._nn.gelu(x_19, approximate="none")
        x_19 = None
        x_21 = torch.nn.functional.dropout(x_20, 0.0, False, False)
        x_20 = None
        x_22 = torch._C._nn.linear(
            x_21,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_21 = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_23 = torch.nn.functional.dropout(x_22, 0.0, False, False)
        x_22 = None
        x_24 = x_18 + x_23
        x_18 = x_23 = None
        layer_norm_4 = torch.nn.functional.layer_norm(
            x_24,
            (448,),
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_norm1_parameters_bias_ = (None)
        linear_8 = torch._C._nn.linear(
            layer_norm_4,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_4 = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_4 = linear_8.reshape(1, 197, 3, 7, 64)
        linear_8 = None
        qkv_2 = reshape_4.permute(2, 0, 3, 1, 4)
        reshape_4 = None
        unbind_2 = qkv_2.unbind(0)
        qkv_2 = None
        q_2 = unbind_2[0]
        k_2 = unbind_2[1]
        v_2 = unbind_2[2]
        unbind_2 = None
        x_25 = torch._C._nn.scaled_dot_product_attention(
            q_2, k_2, v_2, attn_mask=None, dropout_p=0.0
        )
        q_2 = k_2 = v_2 = None
        transpose_4 = x_25.transpose(1, 2)
        x_25 = None
        x_26 = transpose_4.reshape(1, 197, 448)
        transpose_4 = None
        x_27 = torch._C._nn.linear(
            x_26,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_26 = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_28 = torch.nn.functional.dropout(x_27, 0.0, False, False)
        x_27 = None
        x_29 = x_24 + x_28
        x_24 = x_28 = None
        layer_norm_5 = torch.nn.functional.layer_norm(
            x_29,
            (448,),
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_norm2_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_norm2_parameters_bias_ = (None)
        x_30 = torch._C._nn.linear(
            layer_norm_5,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_5 = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_31 = torch._C._nn.gelu(x_30, approximate="none")
        x_30 = None
        x_32 = torch.nn.functional.dropout(x_31, 0.0, False, False)
        x_31 = None
        x_33 = torch._C._nn.linear(
            x_32,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_32 = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_34 = torch.nn.functional.dropout(x_33, 0.0, False, False)
        x_33 = None
        x_35 = x_29 + x_34
        x_29 = x_34 = None
        layer_norm_6 = torch.nn.functional.layer_norm(
            x_35,
            (448,),
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_norm1_parameters_bias_ = (None)
        linear_12 = torch._C._nn.linear(
            layer_norm_6,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_6 = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_6 = linear_12.reshape(1, 197, 3, 7, 64)
        linear_12 = None
        qkv_3 = reshape_6.permute(2, 0, 3, 1, 4)
        reshape_6 = None
        unbind_3 = qkv_3.unbind(0)
        qkv_3 = None
        q_3 = unbind_3[0]
        k_3 = unbind_3[1]
        v_3 = unbind_3[2]
        unbind_3 = None
        x_36 = torch._C._nn.scaled_dot_product_attention(
            q_3, k_3, v_3, attn_mask=None, dropout_p=0.0
        )
        q_3 = k_3 = v_3 = None
        transpose_5 = x_36.transpose(1, 2)
        x_36 = None
        x_37 = transpose_5.reshape(1, 197, 448)
        transpose_5 = None
        x_38 = torch._C._nn.linear(
            x_37,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_bias_,
        )
        x_37 = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_bias_ = (None)
        x_39 = torch.nn.functional.dropout(x_38, 0.0, False, False)
        x_38 = None
        x_40 = x_35 + x_39
        x_35 = x_39 = None
        layer_norm_7 = torch.nn.functional.layer_norm(
            x_40,
            (448,),
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_norm2_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_norm2_parameters_bias_ = (None)
        x_41 = torch._C._nn.linear(
            layer_norm_7,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_7 = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_42 = torch._C._nn.gelu(x_41, approximate="none")
        x_41 = None
        x_43 = torch.nn.functional.dropout(x_42, 0.0, False, False)
        x_42 = None
        x_44 = torch._C._nn.linear(
            x_43,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_43 = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_45 = torch.nn.functional.dropout(x_44, 0.0, False, False)
        x_44 = None
        x_46 = x_40 + x_45
        x_40 = x_45 = None
        layer_norm_8 = torch.nn.functional.layer_norm(
            x_46,
            (448,),
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_norm1_parameters_bias_ = (None)
        linear_16 = torch._C._nn.linear(
            layer_norm_8,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_8 = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_8 = linear_16.reshape(1, 197, 3, 7, 64)
        linear_16 = None
        qkv_4 = reshape_8.permute(2, 0, 3, 1, 4)
        reshape_8 = None
        unbind_4 = qkv_4.unbind(0)
        qkv_4 = None
        q_4 = unbind_4[0]
        k_4 = unbind_4[1]
        v_4 = unbind_4[2]
        unbind_4 = None
        x_47 = torch._C._nn.scaled_dot_product_attention(
            q_4, k_4, v_4, attn_mask=None, dropout_p=0.0
        )
        q_4 = k_4 = v_4 = None
        transpose_6 = x_47.transpose(1, 2)
        x_47 = None
        x_48 = transpose_6.reshape(1, 197, 448)
        transpose_6 = None
        x_49 = torch._C._nn.linear(
            x_48,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_bias_,
        )
        x_48 = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_bias_ = (None)
        x_50 = torch.nn.functional.dropout(x_49, 0.0, False, False)
        x_49 = None
        x_51 = x_46 + x_50
        x_46 = x_50 = None
        layer_norm_9 = torch.nn.functional.layer_norm(
            x_51,
            (448,),
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_norm2_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_norm2_parameters_bias_ = (None)
        x_52 = torch._C._nn.linear(
            layer_norm_9,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_9 = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_53 = torch._C._nn.gelu(x_52, approximate="none")
        x_52 = None
        x_54 = torch.nn.functional.dropout(x_53, 0.0, False, False)
        x_53 = None
        x_55 = torch._C._nn.linear(
            x_54,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_54 = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_56 = torch.nn.functional.dropout(x_55, 0.0, False, False)
        x_55 = None
        x_57 = x_51 + x_56
        x_51 = x_56 = None
        layer_norm_10 = torch.nn.functional.layer_norm(
            x_57,
            (448,),
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_norm1_parameters_bias_ = (None)
        linear_20 = torch._C._nn.linear(
            layer_norm_10,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_10 = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_10 = linear_20.reshape(1, 197, 3, 7, 64)
        linear_20 = None
        qkv_5 = reshape_10.permute(2, 0, 3, 1, 4)
        reshape_10 = None
        unbind_5 = qkv_5.unbind(0)
        qkv_5 = None
        q_5 = unbind_5[0]
        k_5 = unbind_5[1]
        v_5 = unbind_5[2]
        unbind_5 = None
        x_58 = torch._C._nn.scaled_dot_product_attention(
            q_5, k_5, v_5, attn_mask=None, dropout_p=0.0
        )
        q_5 = k_5 = v_5 = None
        transpose_7 = x_58.transpose(1, 2)
        x_58 = None
        x_59 = transpose_7.reshape(1, 197, 448)
        transpose_7 = None
        x_60 = torch._C._nn.linear(
            x_59,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_bias_,
        )
        x_59 = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_bias_ = (None)
        x_61 = torch.nn.functional.dropout(x_60, 0.0, False, False)
        x_60 = None
        x_62 = x_57 + x_61
        x_57 = x_61 = None
        layer_norm_11 = torch.nn.functional.layer_norm(
            x_62,
            (448,),
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_norm2_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_norm2_parameters_bias_ = (None)
        x_63 = torch._C._nn.linear(
            layer_norm_11,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_11 = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_64 = torch._C._nn.gelu(x_63, approximate="none")
        x_63 = None
        x_65 = torch.nn.functional.dropout(x_64, 0.0, False, False)
        x_64 = None
        x_66 = torch._C._nn.linear(
            x_65,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_65 = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_67 = torch.nn.functional.dropout(x_66, 0.0, False, False)
        x_66 = None
        x_68 = x_62 + x_67
        x_62 = x_67 = None
        layer_norm_12 = torch.nn.functional.layer_norm(
            x_68,
            (448,),
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_norm1_parameters_bias_ = (None)
        linear_24 = torch._C._nn.linear(
            layer_norm_12,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_12 = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_12 = linear_24.reshape(1, 197, 3, 7, 64)
        linear_24 = None
        qkv_6 = reshape_12.permute(2, 0, 3, 1, 4)
        reshape_12 = None
        unbind_6 = qkv_6.unbind(0)
        qkv_6 = None
        q_6 = unbind_6[0]
        k_6 = unbind_6[1]
        v_6 = unbind_6[2]
        unbind_6 = None
        x_69 = torch._C._nn.scaled_dot_product_attention(
            q_6, k_6, v_6, attn_mask=None, dropout_p=0.0
        )
        q_6 = k_6 = v_6 = None
        transpose_8 = x_69.transpose(1, 2)
        x_69 = None
        x_70 = transpose_8.reshape(1, 197, 448)
        transpose_8 = None
        x_71 = torch._C._nn.linear(
            x_70,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_attn_modules_proj_parameters_bias_,
        )
        x_70 = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_attn_modules_proj_parameters_bias_ = (None)
        x_72 = torch.nn.functional.dropout(x_71, 0.0, False, False)
        x_71 = None
        x_73 = x_68 + x_72
        x_68 = x_72 = None
        layer_norm_13 = torch.nn.functional.layer_norm(
            x_73,
            (448,),
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_norm2_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_norm2_parameters_bias_ = (None)
        x_74 = torch._C._nn.linear(
            layer_norm_13,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_13 = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_75 = torch._C._nn.gelu(x_74, approximate="none")
        x_74 = None
        x_76 = torch.nn.functional.dropout(x_75, 0.0, False, False)
        x_75 = None
        x_77 = torch._C._nn.linear(
            x_76,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_76 = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_78 = torch.nn.functional.dropout(x_77, 0.0, False, False)
        x_77 = None
        x_79 = x_73 + x_78
        x_73 = x_78 = None
        getitem_37 = x_13[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
        input_1 = torch.nn.functional.layer_norm(
            getitem_37,
            (224,),
            l_self_modules_blocks_modules_0_modules_projs_modules_0_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_projs_modules_0_modules_0_parameters_bias_,
            1e-06,
        )
        getitem_37 = l_self_modules_blocks_modules_0_modules_projs_modules_0_modules_0_parameters_weight_ = l_self_modules_blocks_modules_0_modules_projs_modules_0_modules_0_parameters_bias_ = (None)
        input_2 = torch._C._nn.gelu(input_1, approximate="none")
        input_1 = None
        input_3 = torch._C._nn.linear(
            input_2,
            l_self_modules_blocks_modules_0_modules_projs_modules_0_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_projs_modules_0_modules_2_parameters_bias_,
        )
        input_2 = l_self_modules_blocks_modules_0_modules_projs_modules_0_modules_2_parameters_weight_ = l_self_modules_blocks_modules_0_modules_projs_modules_0_modules_2_parameters_bias_ = (None)
        getitem_38 = x_79[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
        input_4 = torch.nn.functional.layer_norm(
            getitem_38,
            (448,),
            l_self_modules_blocks_modules_0_modules_projs_modules_1_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_projs_modules_1_modules_0_parameters_bias_,
            1e-06,
        )
        getitem_38 = l_self_modules_blocks_modules_0_modules_projs_modules_1_modules_0_parameters_weight_ = l_self_modules_blocks_modules_0_modules_projs_modules_1_modules_0_parameters_bias_ = (None)
        input_5 = torch._C._nn.gelu(input_4, approximate="none")
        input_4 = None
        input_6 = torch._C._nn.linear(
            input_5,
            l_self_modules_blocks_modules_0_modules_projs_modules_1_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_projs_modules_1_modules_2_parameters_bias_,
        )
        input_5 = l_self_modules_blocks_modules_0_modules_projs_modules_1_modules_2_parameters_weight_ = l_self_modules_blocks_modules_0_modules_projs_modules_1_modules_2_parameters_bias_ = (None)
        getitem_39 = x_79[(slice(None, None, None), slice(1, None, None), Ellipsis)]
        tmp = torch.cat((input_3, getitem_39), dim=1)
        input_3 = getitem_39 = None
        getitem_40 = tmp[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
        layer_norm_16 = torch.nn.functional.layer_norm(
            tmp,
            (448,),
            l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_norm1_parameters_bias_,
            1e-06,
        )
        tmp = l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_norm1_parameters_bias_ = (None)
        getitem_41 = layer_norm_16[
            (slice(None, None, None), slice(0, 1, None), Ellipsis)
        ]
        linear_30 = torch._C._nn.linear(
            getitem_41,
            l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wq_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wq_parameters_bias_,
        )
        getitem_41 = l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wq_parameters_weight_ = l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wq_parameters_bias_ = (None)
        reshape_14 = linear_30.reshape(1, 1, 7, 64)
        linear_30 = None
        q_7 = reshape_14.permute(0, 2, 1, 3)
        reshape_14 = None
        linear_31 = torch._C._nn.linear(
            layer_norm_16,
            l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wk_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wk_parameters_bias_,
        )
        l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wk_parameters_weight_ = l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wk_parameters_bias_ = (None)
        reshape_15 = linear_31.reshape(1, 197, 7, 64)
        linear_31 = None
        k_7 = reshape_15.permute(0, 2, 1, 3)
        reshape_15 = None
        linear_32 = torch._C._nn.linear(
            layer_norm_16,
            l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wv_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wv_parameters_bias_,
        )
        layer_norm_16 = l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wv_parameters_weight_ = l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wv_parameters_bias_ = (None)
        reshape_16 = linear_32.reshape(1, 197, 7, 64)
        linear_32 = None
        v_7 = reshape_16.permute(0, 2, 1, 3)
        reshape_16 = None
        transpose_9 = k_7.transpose(-2, -1)
        k_7 = None
        matmul = q_7 @ transpose_9
        q_7 = transpose_9 = None
        attn = matmul * 0.125
        matmul = None
        attn_1 = attn.softmax(dim=-1)
        attn = None
        attn_2 = torch.nn.functional.dropout(attn_1, 0.0, False, False)
        attn_1 = None
        matmul_1 = attn_2 @ v_7
        attn_2 = v_7 = None
        transpose_10 = matmul_1.transpose(1, 2)
        matmul_1 = None
        x_80 = transpose_10.reshape(1, 1, 448)
        transpose_10 = None
        x_81 = torch._C._nn.linear(
            x_80,
            l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_80 = l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_82 = torch.nn.functional.dropout(x_81, 0.0, False, False)
        x_81 = None
        x_83 = getitem_40 + x_82
        getitem_40 = x_82 = None
        getitem_42 = x_83[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
        x_83 = None
        input_7 = torch.nn.functional.layer_norm(
            getitem_42,
            (448,),
            l_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_0_parameters_bias_,
            1e-06,
        )
        getitem_42 = l_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_0_parameters_weight_ = l_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_0_parameters_bias_ = (None)
        input_8 = torch._C._nn.gelu(input_7, approximate="none")
        input_7 = None
        input_9 = torch._C._nn.linear(
            input_8,
            l_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_2_parameters_bias_,
        )
        input_8 = l_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_2_parameters_weight_ = l_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_2_parameters_bias_ = (None)
        getitem_43 = x_13[(slice(None, None, None), slice(1, None, None), Ellipsis)]
        tmp_1 = torch.cat((input_9, getitem_43), dim=1)
        input_9 = getitem_43 = None
        getitem_44 = x_13[(slice(None, None, None), slice(1, None, None), Ellipsis)]
        x_13 = None
        tmp_2 = torch.cat((input_6, getitem_44), dim=1)
        input_6 = getitem_44 = None
        getitem_45 = tmp_2[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
        layer_norm_18 = torch.nn.functional.layer_norm(
            tmp_2,
            (224,),
            l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        tmp_2 = l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_norm1_parameters_bias_ = (None)
        getitem_46 = layer_norm_18[
            (slice(None, None, None), slice(0, 1, None), Ellipsis)
        ]
        linear_35 = torch._C._nn.linear(
            getitem_46,
            l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wq_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wq_parameters_bias_,
        )
        getitem_46 = l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wq_parameters_weight_ = l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wq_parameters_bias_ = (None)
        reshape_18 = linear_35.reshape(1, 1, 7, 32)
        linear_35 = None
        q_8 = reshape_18.permute(0, 2, 1, 3)
        reshape_18 = None
        linear_36 = torch._C._nn.linear(
            layer_norm_18,
            l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wk_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wk_parameters_bias_,
        )
        l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wk_parameters_weight_ = l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wk_parameters_bias_ = (None)
        reshape_19 = linear_36.reshape(1, 401, 7, 32)
        linear_36 = None
        k_8 = reshape_19.permute(0, 2, 1, 3)
        reshape_19 = None
        linear_37 = torch._C._nn.linear(
            layer_norm_18,
            l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wv_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wv_parameters_bias_,
        )
        layer_norm_18 = l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wv_parameters_weight_ = l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wv_parameters_bias_ = (None)
        reshape_20 = linear_37.reshape(1, 401, 7, 32)
        linear_37 = None
        v_8 = reshape_20.permute(0, 2, 1, 3)
        reshape_20 = None
        transpose_11 = k_8.transpose(-2, -1)
        k_8 = None
        matmul_2 = q_8 @ transpose_11
        q_8 = transpose_11 = None
        attn_3 = matmul_2 * 0.1767766952966369
        matmul_2 = None
        attn_4 = attn_3.softmax(dim=-1)
        attn_3 = None
        attn_5 = torch.nn.functional.dropout(attn_4, 0.0, False, False)
        attn_4 = None
        matmul_3 = attn_5 @ v_8
        attn_5 = v_8 = None
        transpose_12 = matmul_3.transpose(1, 2)
        matmul_3 = None
        x_84 = transpose_12.reshape(1, 1, 224)
        transpose_12 = None
        x_85 = torch._C._nn.linear(
            x_84,
            l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_84 = l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_86 = torch.nn.functional.dropout(x_85, 0.0, False, False)
        x_85 = None
        x_87 = getitem_45 + x_86
        getitem_45 = x_86 = None
        getitem_47 = x_87[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
        x_87 = None
        input_10 = torch.nn.functional.layer_norm(
            getitem_47,
            (224,),
            l_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_0_parameters_bias_,
            1e-06,
        )
        getitem_47 = l_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_0_parameters_weight_ = l_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_0_parameters_bias_ = (None)
        input_11 = torch._C._nn.gelu(input_10, approximate="none")
        input_10 = None
        input_12 = torch._C._nn.linear(
            input_11,
            l_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_2_parameters_bias_,
        )
        input_11 = l_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_2_parameters_weight_ = l_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_2_parameters_bias_ = (None)
        getitem_48 = x_79[(slice(None, None, None), slice(1, None, None), Ellipsis)]
        x_79 = None
        tmp_3 = torch.cat((input_12, getitem_48), dim=1)
        input_12 = getitem_48 = None
        layer_norm_20 = torch.nn.functional.layer_norm(
            tmp_1,
            (224,),
            l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_ = (None)
        linear_40 = torch._C._nn.linear(
            layer_norm_20,
            l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_20 = l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_22 = linear_40.reshape(1, 401, 3, 7, 32)
        linear_40 = None
        qkv_7 = reshape_22.permute(2, 0, 3, 1, 4)
        reshape_22 = None
        unbind_7 = qkv_7.unbind(0)
        qkv_7 = None
        q_9 = unbind_7[0]
        k_9 = unbind_7[1]
        v_9 = unbind_7[2]
        unbind_7 = None
        x_88 = torch._C._nn.scaled_dot_product_attention(
            q_9, k_9, v_9, attn_mask=None, dropout_p=0.0
        )
        q_9 = k_9 = v_9 = None
        transpose_13 = x_88.transpose(1, 2)
        x_88 = None
        x_89 = transpose_13.reshape(1, 401, 224)
        transpose_13 = None
        x_90 = torch._C._nn.linear(
            x_89,
            l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_89 = l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_91 = torch.nn.functional.dropout(x_90, 0.0, False, False)
        x_90 = None
        x_92 = tmp_1 + x_91
        tmp_1 = x_91 = None
        layer_norm_21 = torch.nn.functional.layer_norm(
            x_92,
            (224,),
            l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_ = (None)
        x_93 = torch._C._nn.linear(
            layer_norm_21,
            l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_21 = l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_94 = torch._C._nn.gelu(x_93, approximate="none")
        x_93 = None
        x_95 = torch.nn.functional.dropout(x_94, 0.0, False, False)
        x_94 = None
        x_96 = torch._C._nn.linear(
            x_95,
            l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_95 = l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_97 = torch.nn.functional.dropout(x_96, 0.0, False, False)
        x_96 = None
        x_98 = x_92 + x_97
        x_92 = x_97 = None
        layer_norm_22 = torch.nn.functional.layer_norm(
            tmp_3,
            (448,),
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm1_parameters_bias_ = (None)
        linear_44 = torch._C._nn.linear(
            layer_norm_22,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_22 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_24 = linear_44.reshape(1, 197, 3, 7, 64)
        linear_44 = None
        qkv_8 = reshape_24.permute(2, 0, 3, 1, 4)
        reshape_24 = None
        unbind_8 = qkv_8.unbind(0)
        qkv_8 = None
        q_10 = unbind_8[0]
        k_10 = unbind_8[1]
        v_10 = unbind_8[2]
        unbind_8 = None
        x_99 = torch._C._nn.scaled_dot_product_attention(
            q_10, k_10, v_10, attn_mask=None, dropout_p=0.0
        )
        q_10 = k_10 = v_10 = None
        transpose_14 = x_99.transpose(1, 2)
        x_99 = None
        x_100 = transpose_14.reshape(1, 197, 448)
        transpose_14 = None
        x_101 = torch._C._nn.linear(
            x_100,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_100 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_102 = torch.nn.functional.dropout(x_101, 0.0, False, False)
        x_101 = None
        x_103 = tmp_3 + x_102
        tmp_3 = x_102 = None
        layer_norm_23 = torch.nn.functional.layer_norm(
            x_103,
            (448,),
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm2_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm2_parameters_bias_ = (None)
        x_104 = torch._C._nn.linear(
            layer_norm_23,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_23 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_105 = torch._C._nn.gelu(x_104, approximate="none")
        x_104 = None
        x_106 = torch.nn.functional.dropout(x_105, 0.0, False, False)
        x_105 = None
        x_107 = torch._C._nn.linear(
            x_106,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_106 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_108 = torch.nn.functional.dropout(x_107, 0.0, False, False)
        x_107 = None
        x_109 = x_103 + x_108
        x_103 = x_108 = None
        layer_norm_24 = torch.nn.functional.layer_norm(
            x_109,
            (448,),
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm1_parameters_bias_ = (None)
        linear_48 = torch._C._nn.linear(
            layer_norm_24,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_24 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_26 = linear_48.reshape(1, 197, 3, 7, 64)
        linear_48 = None
        qkv_9 = reshape_26.permute(2, 0, 3, 1, 4)
        reshape_26 = None
        unbind_9 = qkv_9.unbind(0)
        qkv_9 = None
        q_11 = unbind_9[0]
        k_11 = unbind_9[1]
        v_11 = unbind_9[2]
        unbind_9 = None
        x_110 = torch._C._nn.scaled_dot_product_attention(
            q_11, k_11, v_11, attn_mask=None, dropout_p=0.0
        )
        q_11 = k_11 = v_11 = None
        transpose_15 = x_110.transpose(1, 2)
        x_110 = None
        x_111 = transpose_15.reshape(1, 197, 448)
        transpose_15 = None
        x_112 = torch._C._nn.linear(
            x_111,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_111 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_113 = torch.nn.functional.dropout(x_112, 0.0, False, False)
        x_112 = None
        x_114 = x_109 + x_113
        x_109 = x_113 = None
        layer_norm_25 = torch.nn.functional.layer_norm(
            x_114,
            (448,),
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm2_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm2_parameters_bias_ = (None)
        x_115 = torch._C._nn.linear(
            layer_norm_25,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_25 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_116 = torch._C._nn.gelu(x_115, approximate="none")
        x_115 = None
        x_117 = torch.nn.functional.dropout(x_116, 0.0, False, False)
        x_116 = None
        x_118 = torch._C._nn.linear(
            x_117,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_117 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_119 = torch.nn.functional.dropout(x_118, 0.0, False, False)
        x_118 = None
        x_120 = x_114 + x_119
        x_114 = x_119 = None
        layer_norm_26 = torch.nn.functional.layer_norm(
            x_120,
            (448,),
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm1_parameters_bias_ = (None)
        linear_52 = torch._C._nn.linear(
            layer_norm_26,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_26 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_28 = linear_52.reshape(1, 197, 3, 7, 64)
        linear_52 = None
        qkv_10 = reshape_28.permute(2, 0, 3, 1, 4)
        reshape_28 = None
        unbind_10 = qkv_10.unbind(0)
        qkv_10 = None
        q_12 = unbind_10[0]
        k_12 = unbind_10[1]
        v_12 = unbind_10[2]
        unbind_10 = None
        x_121 = torch._C._nn.scaled_dot_product_attention(
            q_12, k_12, v_12, attn_mask=None, dropout_p=0.0
        )
        q_12 = k_12 = v_12 = None
        transpose_16 = x_121.transpose(1, 2)
        x_121 = None
        x_122 = transpose_16.reshape(1, 197, 448)
        transpose_16 = None
        x_123 = torch._C._nn.linear(
            x_122,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_bias_,
        )
        x_122 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_bias_ = (None)
        x_124 = torch.nn.functional.dropout(x_123, 0.0, False, False)
        x_123 = None
        x_125 = x_120 + x_124
        x_120 = x_124 = None
        layer_norm_27 = torch.nn.functional.layer_norm(
            x_125,
            (448,),
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm2_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm2_parameters_bias_ = (None)
        x_126 = torch._C._nn.linear(
            layer_norm_27,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_27 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_127 = torch._C._nn.gelu(x_126, approximate="none")
        x_126 = None
        x_128 = torch.nn.functional.dropout(x_127, 0.0, False, False)
        x_127 = None
        x_129 = torch._C._nn.linear(
            x_128,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_128 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_130 = torch.nn.functional.dropout(x_129, 0.0, False, False)
        x_129 = None
        x_131 = x_125 + x_130
        x_125 = x_130 = None
        layer_norm_28 = torch.nn.functional.layer_norm(
            x_131,
            (448,),
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_norm1_parameters_bias_ = (None)
        linear_56 = torch._C._nn.linear(
            layer_norm_28,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_28 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_30 = linear_56.reshape(1, 197, 3, 7, 64)
        linear_56 = None
        qkv_11 = reshape_30.permute(2, 0, 3, 1, 4)
        reshape_30 = None
        unbind_11 = qkv_11.unbind(0)
        qkv_11 = None
        q_13 = unbind_11[0]
        k_13 = unbind_11[1]
        v_13 = unbind_11[2]
        unbind_11 = None
        x_132 = torch._C._nn.scaled_dot_product_attention(
            q_13, k_13, v_13, attn_mask=None, dropout_p=0.0
        )
        q_13 = k_13 = v_13 = None
        transpose_17 = x_132.transpose(1, 2)
        x_132 = None
        x_133 = transpose_17.reshape(1, 197, 448)
        transpose_17 = None
        x_134 = torch._C._nn.linear(
            x_133,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_bias_,
        )
        x_133 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_bias_ = (None)
        x_135 = torch.nn.functional.dropout(x_134, 0.0, False, False)
        x_134 = None
        x_136 = x_131 + x_135
        x_131 = x_135 = None
        layer_norm_29 = torch.nn.functional.layer_norm(
            x_136,
            (448,),
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_norm2_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_norm2_parameters_bias_ = (None)
        x_137 = torch._C._nn.linear(
            layer_norm_29,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_29 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_138 = torch._C._nn.gelu(x_137, approximate="none")
        x_137 = None
        x_139 = torch.nn.functional.dropout(x_138, 0.0, False, False)
        x_138 = None
        x_140 = torch._C._nn.linear(
            x_139,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_139 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_141 = torch.nn.functional.dropout(x_140, 0.0, False, False)
        x_140 = None
        x_142 = x_136 + x_141
        x_136 = x_141 = None
        layer_norm_30 = torch.nn.functional.layer_norm(
            x_142,
            (448,),
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_norm1_parameters_bias_ = (None)
        linear_60 = torch._C._nn.linear(
            layer_norm_30,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_30 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_32 = linear_60.reshape(1, 197, 3, 7, 64)
        linear_60 = None
        qkv_12 = reshape_32.permute(2, 0, 3, 1, 4)
        reshape_32 = None
        unbind_12 = qkv_12.unbind(0)
        qkv_12 = None
        q_14 = unbind_12[0]
        k_14 = unbind_12[1]
        v_14 = unbind_12[2]
        unbind_12 = None
        x_143 = torch._C._nn.scaled_dot_product_attention(
            q_14, k_14, v_14, attn_mask=None, dropout_p=0.0
        )
        q_14 = k_14 = v_14 = None
        transpose_18 = x_143.transpose(1, 2)
        x_143 = None
        x_144 = transpose_18.reshape(1, 197, 448)
        transpose_18 = None
        x_145 = torch._C._nn.linear(
            x_144,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_bias_,
        )
        x_144 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_bias_ = (None)
        x_146 = torch.nn.functional.dropout(x_145, 0.0, False, False)
        x_145 = None
        x_147 = x_142 + x_146
        x_142 = x_146 = None
        layer_norm_31 = torch.nn.functional.layer_norm(
            x_147,
            (448,),
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_norm2_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_norm2_parameters_bias_ = (None)
        x_148 = torch._C._nn.linear(
            layer_norm_31,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_31 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_149 = torch._C._nn.gelu(x_148, approximate="none")
        x_148 = None
        x_150 = torch.nn.functional.dropout(x_149, 0.0, False, False)
        x_149 = None
        x_151 = torch._C._nn.linear(
            x_150,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_150 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_152 = torch.nn.functional.dropout(x_151, 0.0, False, False)
        x_151 = None
        x_153 = x_147 + x_152
        x_147 = x_152 = None
        layer_norm_32 = torch.nn.functional.layer_norm(
            x_153,
            (448,),
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_norm1_parameters_bias_ = (None)
        linear_64 = torch._C._nn.linear(
            layer_norm_32,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_32 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_34 = linear_64.reshape(1, 197, 3, 7, 64)
        linear_64 = None
        qkv_13 = reshape_34.permute(2, 0, 3, 1, 4)
        reshape_34 = None
        unbind_13 = qkv_13.unbind(0)
        qkv_13 = None
        q_15 = unbind_13[0]
        k_15 = unbind_13[1]
        v_15 = unbind_13[2]
        unbind_13 = None
        x_154 = torch._C._nn.scaled_dot_product_attention(
            q_15, k_15, v_15, attn_mask=None, dropout_p=0.0
        )
        q_15 = k_15 = v_15 = None
        transpose_19 = x_154.transpose(1, 2)
        x_154 = None
        x_155 = transpose_19.reshape(1, 197, 448)
        transpose_19 = None
        x_156 = torch._C._nn.linear(
            x_155,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_attn_modules_proj_parameters_bias_,
        )
        x_155 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_attn_modules_proj_parameters_bias_ = (None)
        x_157 = torch.nn.functional.dropout(x_156, 0.0, False, False)
        x_156 = None
        x_158 = x_153 + x_157
        x_153 = x_157 = None
        layer_norm_33 = torch.nn.functional.layer_norm(
            x_158,
            (448,),
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_norm2_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_norm2_parameters_bias_ = (None)
        x_159 = torch._C._nn.linear(
            layer_norm_33,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_33 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_160 = torch._C._nn.gelu(x_159, approximate="none")
        x_159 = None
        x_161 = torch.nn.functional.dropout(x_160, 0.0, False, False)
        x_160 = None
        x_162 = torch._C._nn.linear(
            x_161,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_161 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_163 = torch.nn.functional.dropout(x_162, 0.0, False, False)
        x_162 = None
        x_164 = x_158 + x_163
        x_158 = x_163 = None
        getitem_70 = x_98[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
        input_13 = torch.nn.functional.layer_norm(
            getitem_70,
            (224,),
            l_self_modules_blocks_modules_1_modules_projs_modules_0_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_projs_modules_0_modules_0_parameters_bias_,
            1e-06,
        )
        getitem_70 = l_self_modules_blocks_modules_1_modules_projs_modules_0_modules_0_parameters_weight_ = l_self_modules_blocks_modules_1_modules_projs_modules_0_modules_0_parameters_bias_ = (None)
        input_14 = torch._C._nn.gelu(input_13, approximate="none")
        input_13 = None
        input_15 = torch._C._nn.linear(
            input_14,
            l_self_modules_blocks_modules_1_modules_projs_modules_0_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_projs_modules_0_modules_2_parameters_bias_,
        )
        input_14 = l_self_modules_blocks_modules_1_modules_projs_modules_0_modules_2_parameters_weight_ = l_self_modules_blocks_modules_1_modules_projs_modules_0_modules_2_parameters_bias_ = (None)
        getitem_71 = x_164[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
        input_16 = torch.nn.functional.layer_norm(
            getitem_71,
            (448,),
            l_self_modules_blocks_modules_1_modules_projs_modules_1_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_projs_modules_1_modules_0_parameters_bias_,
            1e-06,
        )
        getitem_71 = l_self_modules_blocks_modules_1_modules_projs_modules_1_modules_0_parameters_weight_ = l_self_modules_blocks_modules_1_modules_projs_modules_1_modules_0_parameters_bias_ = (None)
        input_17 = torch._C._nn.gelu(input_16, approximate="none")
        input_16 = None
        input_18 = torch._C._nn.linear(
            input_17,
            l_self_modules_blocks_modules_1_modules_projs_modules_1_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_projs_modules_1_modules_2_parameters_bias_,
        )
        input_17 = l_self_modules_blocks_modules_1_modules_projs_modules_1_modules_2_parameters_weight_ = l_self_modules_blocks_modules_1_modules_projs_modules_1_modules_2_parameters_bias_ = (None)
        getitem_72 = x_164[(slice(None, None, None), slice(1, None, None), Ellipsis)]
        tmp_4 = torch.cat((input_15, getitem_72), dim=1)
        input_15 = getitem_72 = None
        getitem_73 = tmp_4[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
        layer_norm_36 = torch.nn.functional.layer_norm(
            tmp_4,
            (448,),
            l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_norm1_parameters_bias_,
            1e-06,
        )
        tmp_4 = l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_norm1_parameters_bias_ = (None)
        getitem_74 = layer_norm_36[
            (slice(None, None, None), slice(0, 1, None), Ellipsis)
        ]
        linear_70 = torch._C._nn.linear(
            getitem_74,
            l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wq_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wq_parameters_bias_,
        )
        getitem_74 = l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wq_parameters_weight_ = l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wq_parameters_bias_ = (None)
        reshape_36 = linear_70.reshape(1, 1, 7, 64)
        linear_70 = None
        q_16 = reshape_36.permute(0, 2, 1, 3)
        reshape_36 = None
        linear_71 = torch._C._nn.linear(
            layer_norm_36,
            l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wk_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wk_parameters_bias_,
        )
        l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wk_parameters_weight_ = l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wk_parameters_bias_ = (None)
        reshape_37 = linear_71.reshape(1, 197, 7, 64)
        linear_71 = None
        k_16 = reshape_37.permute(0, 2, 1, 3)
        reshape_37 = None
        linear_72 = torch._C._nn.linear(
            layer_norm_36,
            l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wv_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wv_parameters_bias_,
        )
        layer_norm_36 = l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wv_parameters_weight_ = l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wv_parameters_bias_ = (None)
        reshape_38 = linear_72.reshape(1, 197, 7, 64)
        linear_72 = None
        v_16 = reshape_38.permute(0, 2, 1, 3)
        reshape_38 = None
        transpose_20 = k_16.transpose(-2, -1)
        k_16 = None
        matmul_4 = q_16 @ transpose_20
        q_16 = transpose_20 = None
        attn_6 = matmul_4 * 0.125
        matmul_4 = None
        attn_7 = attn_6.softmax(dim=-1)
        attn_6 = None
        attn_8 = torch.nn.functional.dropout(attn_7, 0.0, False, False)
        attn_7 = None
        matmul_5 = attn_8 @ v_16
        attn_8 = v_16 = None
        transpose_21 = matmul_5.transpose(1, 2)
        matmul_5 = None
        x_165 = transpose_21.reshape(1, 1, 448)
        transpose_21 = None
        x_166 = torch._C._nn.linear(
            x_165,
            l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_165 = l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_167 = torch.nn.functional.dropout(x_166, 0.0, False, False)
        x_166 = None
        x_168 = getitem_73 + x_167
        getitem_73 = x_167 = None
        getitem_75 = x_168[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
        x_168 = None
        input_19 = torch.nn.functional.layer_norm(
            getitem_75,
            (448,),
            l_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_0_parameters_bias_,
            1e-06,
        )
        getitem_75 = l_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_0_parameters_weight_ = l_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_0_parameters_bias_ = (None)
        input_20 = torch._C._nn.gelu(input_19, approximate="none")
        input_19 = None
        input_21 = torch._C._nn.linear(
            input_20,
            l_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_2_parameters_bias_,
        )
        input_20 = l_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_2_parameters_weight_ = l_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_2_parameters_bias_ = (None)
        getitem_76 = x_98[(slice(None, None, None), slice(1, None, None), Ellipsis)]
        tmp_5 = torch.cat((input_21, getitem_76), dim=1)
        input_21 = getitem_76 = None
        getitem_77 = x_98[(slice(None, None, None), slice(1, None, None), Ellipsis)]
        x_98 = None
        tmp_6 = torch.cat((input_18, getitem_77), dim=1)
        input_18 = getitem_77 = None
        getitem_78 = tmp_6[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
        layer_norm_38 = torch.nn.functional.layer_norm(
            tmp_6,
            (224,),
            l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        tmp_6 = l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_norm1_parameters_bias_ = (None)
        getitem_79 = layer_norm_38[
            (slice(None, None, None), slice(0, 1, None), Ellipsis)
        ]
        linear_75 = torch._C._nn.linear(
            getitem_79,
            l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wq_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wq_parameters_bias_,
        )
        getitem_79 = l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wq_parameters_weight_ = l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wq_parameters_bias_ = (None)
        reshape_40 = linear_75.reshape(1, 1, 7, 32)
        linear_75 = None
        q_17 = reshape_40.permute(0, 2, 1, 3)
        reshape_40 = None
        linear_76 = torch._C._nn.linear(
            layer_norm_38,
            l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wk_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wk_parameters_bias_,
        )
        l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wk_parameters_weight_ = l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wk_parameters_bias_ = (None)
        reshape_41 = linear_76.reshape(1, 401, 7, 32)
        linear_76 = None
        k_17 = reshape_41.permute(0, 2, 1, 3)
        reshape_41 = None
        linear_77 = torch._C._nn.linear(
            layer_norm_38,
            l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wv_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wv_parameters_bias_,
        )
        layer_norm_38 = l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wv_parameters_weight_ = l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wv_parameters_bias_ = (None)
        reshape_42 = linear_77.reshape(1, 401, 7, 32)
        linear_77 = None
        v_17 = reshape_42.permute(0, 2, 1, 3)
        reshape_42 = None
        transpose_22 = k_17.transpose(-2, -1)
        k_17 = None
        matmul_6 = q_17 @ transpose_22
        q_17 = transpose_22 = None
        attn_9 = matmul_6 * 0.1767766952966369
        matmul_6 = None
        attn_10 = attn_9.softmax(dim=-1)
        attn_9 = None
        attn_11 = torch.nn.functional.dropout(attn_10, 0.0, False, False)
        attn_10 = None
        matmul_7 = attn_11 @ v_17
        attn_11 = v_17 = None
        transpose_23 = matmul_7.transpose(1, 2)
        matmul_7 = None
        x_169 = transpose_23.reshape(1, 1, 224)
        transpose_23 = None
        x_170 = torch._C._nn.linear(
            x_169,
            l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_169 = l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_171 = torch.nn.functional.dropout(x_170, 0.0, False, False)
        x_170 = None
        x_172 = getitem_78 + x_171
        getitem_78 = x_171 = None
        getitem_80 = x_172[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
        x_172 = None
        input_22 = torch.nn.functional.layer_norm(
            getitem_80,
            (224,),
            l_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_0_parameters_bias_,
            1e-06,
        )
        getitem_80 = l_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_0_parameters_weight_ = l_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_0_parameters_bias_ = (None)
        input_23 = torch._C._nn.gelu(input_22, approximate="none")
        input_22 = None
        input_24 = torch._C._nn.linear(
            input_23,
            l_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_2_parameters_bias_,
        )
        input_23 = l_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_2_parameters_weight_ = l_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_2_parameters_bias_ = (None)
        getitem_81 = x_164[(slice(None, None, None), slice(1, None, None), Ellipsis)]
        x_164 = None
        tmp_7 = torch.cat((input_24, getitem_81), dim=1)
        input_24 = getitem_81 = None
        layer_norm_40 = torch.nn.functional.layer_norm(
            tmp_5,
            (224,),
            l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_ = (None)
        linear_80 = torch._C._nn.linear(
            layer_norm_40,
            l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_40 = l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_44 = linear_80.reshape(1, 401, 3, 7, 32)
        linear_80 = None
        qkv_14 = reshape_44.permute(2, 0, 3, 1, 4)
        reshape_44 = None
        unbind_14 = qkv_14.unbind(0)
        qkv_14 = None
        q_18 = unbind_14[0]
        k_18 = unbind_14[1]
        v_18 = unbind_14[2]
        unbind_14 = None
        x_173 = torch._C._nn.scaled_dot_product_attention(
            q_18, k_18, v_18, attn_mask=None, dropout_p=0.0
        )
        q_18 = k_18 = v_18 = None
        transpose_24 = x_173.transpose(1, 2)
        x_173 = None
        x_174 = transpose_24.reshape(1, 401, 224)
        transpose_24 = None
        x_175 = torch._C._nn.linear(
            x_174,
            l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_174 = l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_176 = torch.nn.functional.dropout(x_175, 0.0, False, False)
        x_175 = None
        x_177 = tmp_5 + x_176
        tmp_5 = x_176 = None
        layer_norm_41 = torch.nn.functional.layer_norm(
            x_177,
            (224,),
            l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_ = (None)
        x_178 = torch._C._nn.linear(
            layer_norm_41,
            l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_41 = l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_179 = torch._C._nn.gelu(x_178, approximate="none")
        x_178 = None
        x_180 = torch.nn.functional.dropout(x_179, 0.0, False, False)
        x_179 = None
        x_181 = torch._C._nn.linear(
            x_180,
            l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_180 = l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_182 = torch.nn.functional.dropout(x_181, 0.0, False, False)
        x_181 = None
        x_183 = x_177 + x_182
        x_177 = x_182 = None
        layer_norm_42 = torch.nn.functional.layer_norm(
            tmp_7,
            (448,),
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm1_parameters_bias_ = (None)
        linear_84 = torch._C._nn.linear(
            layer_norm_42,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_42 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_46 = linear_84.reshape(1, 197, 3, 7, 64)
        linear_84 = None
        qkv_15 = reshape_46.permute(2, 0, 3, 1, 4)
        reshape_46 = None
        unbind_15 = qkv_15.unbind(0)
        qkv_15 = None
        q_19 = unbind_15[0]
        k_19 = unbind_15[1]
        v_19 = unbind_15[2]
        unbind_15 = None
        x_184 = torch._C._nn.scaled_dot_product_attention(
            q_19, k_19, v_19, attn_mask=None, dropout_p=0.0
        )
        q_19 = k_19 = v_19 = None
        transpose_25 = x_184.transpose(1, 2)
        x_184 = None
        x_185 = transpose_25.reshape(1, 197, 448)
        transpose_25 = None
        x_186 = torch._C._nn.linear(
            x_185,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_185 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_187 = torch.nn.functional.dropout(x_186, 0.0, False, False)
        x_186 = None
        x_188 = tmp_7 + x_187
        tmp_7 = x_187 = None
        layer_norm_43 = torch.nn.functional.layer_norm(
            x_188,
            (448,),
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm2_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm2_parameters_bias_ = (None)
        x_189 = torch._C._nn.linear(
            layer_norm_43,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_43 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_190 = torch._C._nn.gelu(x_189, approximate="none")
        x_189 = None
        x_191 = torch.nn.functional.dropout(x_190, 0.0, False, False)
        x_190 = None
        x_192 = torch._C._nn.linear(
            x_191,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_191 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_193 = torch.nn.functional.dropout(x_192, 0.0, False, False)
        x_192 = None
        x_194 = x_188 + x_193
        x_188 = x_193 = None
        layer_norm_44 = torch.nn.functional.layer_norm(
            x_194,
            (448,),
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm1_parameters_bias_ = (None)
        linear_88 = torch._C._nn.linear(
            layer_norm_44,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_44 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_48 = linear_88.reshape(1, 197, 3, 7, 64)
        linear_88 = None
        qkv_16 = reshape_48.permute(2, 0, 3, 1, 4)
        reshape_48 = None
        unbind_16 = qkv_16.unbind(0)
        qkv_16 = None
        q_20 = unbind_16[0]
        k_20 = unbind_16[1]
        v_20 = unbind_16[2]
        unbind_16 = None
        x_195 = torch._C._nn.scaled_dot_product_attention(
            q_20, k_20, v_20, attn_mask=None, dropout_p=0.0
        )
        q_20 = k_20 = v_20 = None
        transpose_26 = x_195.transpose(1, 2)
        x_195 = None
        x_196 = transpose_26.reshape(1, 197, 448)
        transpose_26 = None
        x_197 = torch._C._nn.linear(
            x_196,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_196 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_198 = torch.nn.functional.dropout(x_197, 0.0, False, False)
        x_197 = None
        x_199 = x_194 + x_198
        x_194 = x_198 = None
        layer_norm_45 = torch.nn.functional.layer_norm(
            x_199,
            (448,),
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm2_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm2_parameters_bias_ = (None)
        x_200 = torch._C._nn.linear(
            layer_norm_45,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_45 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_201 = torch._C._nn.gelu(x_200, approximate="none")
        x_200 = None
        x_202 = torch.nn.functional.dropout(x_201, 0.0, False, False)
        x_201 = None
        x_203 = torch._C._nn.linear(
            x_202,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_202 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_204 = torch.nn.functional.dropout(x_203, 0.0, False, False)
        x_203 = None
        x_205 = x_199 + x_204
        x_199 = x_204 = None
        layer_norm_46 = torch.nn.functional.layer_norm(
            x_205,
            (448,),
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm1_parameters_bias_ = (None)
        linear_92 = torch._C._nn.linear(
            layer_norm_46,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_46 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_50 = linear_92.reshape(1, 197, 3, 7, 64)
        linear_92 = None
        qkv_17 = reshape_50.permute(2, 0, 3, 1, 4)
        reshape_50 = None
        unbind_17 = qkv_17.unbind(0)
        qkv_17 = None
        q_21 = unbind_17[0]
        k_21 = unbind_17[1]
        v_21 = unbind_17[2]
        unbind_17 = None
        x_206 = torch._C._nn.scaled_dot_product_attention(
            q_21, k_21, v_21, attn_mask=None, dropout_p=0.0
        )
        q_21 = k_21 = v_21 = None
        transpose_27 = x_206.transpose(1, 2)
        x_206 = None
        x_207 = transpose_27.reshape(1, 197, 448)
        transpose_27 = None
        x_208 = torch._C._nn.linear(
            x_207,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_bias_,
        )
        x_207 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_bias_ = (None)
        x_209 = torch.nn.functional.dropout(x_208, 0.0, False, False)
        x_208 = None
        x_210 = x_205 + x_209
        x_205 = x_209 = None
        layer_norm_47 = torch.nn.functional.layer_norm(
            x_210,
            (448,),
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm2_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm2_parameters_bias_ = (None)
        x_211 = torch._C._nn.linear(
            layer_norm_47,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_47 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_212 = torch._C._nn.gelu(x_211, approximate="none")
        x_211 = None
        x_213 = torch.nn.functional.dropout(x_212, 0.0, False, False)
        x_212 = None
        x_214 = torch._C._nn.linear(
            x_213,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_213 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_215 = torch.nn.functional.dropout(x_214, 0.0, False, False)
        x_214 = None
        x_216 = x_210 + x_215
        x_210 = x_215 = None
        layer_norm_48 = torch.nn.functional.layer_norm(
            x_216,
            (448,),
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_norm1_parameters_bias_ = (None)
        linear_96 = torch._C._nn.linear(
            layer_norm_48,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_48 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_52 = linear_96.reshape(1, 197, 3, 7, 64)
        linear_96 = None
        qkv_18 = reshape_52.permute(2, 0, 3, 1, 4)
        reshape_52 = None
        unbind_18 = qkv_18.unbind(0)
        qkv_18 = None
        q_22 = unbind_18[0]
        k_22 = unbind_18[1]
        v_22 = unbind_18[2]
        unbind_18 = None
        x_217 = torch._C._nn.scaled_dot_product_attention(
            q_22, k_22, v_22, attn_mask=None, dropout_p=0.0
        )
        q_22 = k_22 = v_22 = None
        transpose_28 = x_217.transpose(1, 2)
        x_217 = None
        x_218 = transpose_28.reshape(1, 197, 448)
        transpose_28 = None
        x_219 = torch._C._nn.linear(
            x_218,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_bias_,
        )
        x_218 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_bias_ = (None)
        x_220 = torch.nn.functional.dropout(x_219, 0.0, False, False)
        x_219 = None
        x_221 = x_216 + x_220
        x_216 = x_220 = None
        layer_norm_49 = torch.nn.functional.layer_norm(
            x_221,
            (448,),
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_norm2_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_norm2_parameters_bias_ = (None)
        x_222 = torch._C._nn.linear(
            layer_norm_49,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_49 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_223 = torch._C._nn.gelu(x_222, approximate="none")
        x_222 = None
        x_224 = torch.nn.functional.dropout(x_223, 0.0, False, False)
        x_223 = None
        x_225 = torch._C._nn.linear(
            x_224,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_224 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_226 = torch.nn.functional.dropout(x_225, 0.0, False, False)
        x_225 = None
        x_227 = x_221 + x_226
        x_221 = x_226 = None
        layer_norm_50 = torch.nn.functional.layer_norm(
            x_227,
            (448,),
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_norm1_parameters_bias_ = (None)
        linear_100 = torch._C._nn.linear(
            layer_norm_50,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_50 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_54 = linear_100.reshape(1, 197, 3, 7, 64)
        linear_100 = None
        qkv_19 = reshape_54.permute(2, 0, 3, 1, 4)
        reshape_54 = None
        unbind_19 = qkv_19.unbind(0)
        qkv_19 = None
        q_23 = unbind_19[0]
        k_23 = unbind_19[1]
        v_23 = unbind_19[2]
        unbind_19 = None
        x_228 = torch._C._nn.scaled_dot_product_attention(
            q_23, k_23, v_23, attn_mask=None, dropout_p=0.0
        )
        q_23 = k_23 = v_23 = None
        transpose_29 = x_228.transpose(1, 2)
        x_228 = None
        x_229 = transpose_29.reshape(1, 197, 448)
        transpose_29 = None
        x_230 = torch._C._nn.linear(
            x_229,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_bias_,
        )
        x_229 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_bias_ = (None)
        x_231 = torch.nn.functional.dropout(x_230, 0.0, False, False)
        x_230 = None
        x_232 = x_227 + x_231
        x_227 = x_231 = None
        layer_norm_51 = torch.nn.functional.layer_norm(
            x_232,
            (448,),
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_norm2_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_norm2_parameters_bias_ = (None)
        x_233 = torch._C._nn.linear(
            layer_norm_51,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_51 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_234 = torch._C._nn.gelu(x_233, approximate="none")
        x_233 = None
        x_235 = torch.nn.functional.dropout(x_234, 0.0, False, False)
        x_234 = None
        x_236 = torch._C._nn.linear(
            x_235,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_235 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_237 = torch.nn.functional.dropout(x_236, 0.0, False, False)
        x_236 = None
        x_238 = x_232 + x_237
        x_232 = x_237 = None
        layer_norm_52 = torch.nn.functional.layer_norm(
            x_238,
            (448,),
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_norm1_parameters_bias_ = (None)
        linear_104 = torch._C._nn.linear(
            layer_norm_52,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_52 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_56 = linear_104.reshape(1, 197, 3, 7, 64)
        linear_104 = None
        qkv_20 = reshape_56.permute(2, 0, 3, 1, 4)
        reshape_56 = None
        unbind_20 = qkv_20.unbind(0)
        qkv_20 = None
        q_24 = unbind_20[0]
        k_24 = unbind_20[1]
        v_24 = unbind_20[2]
        unbind_20 = None
        x_239 = torch._C._nn.scaled_dot_product_attention(
            q_24, k_24, v_24, attn_mask=None, dropout_p=0.0
        )
        q_24 = k_24 = v_24 = None
        transpose_30 = x_239.transpose(1, 2)
        x_239 = None
        x_240 = transpose_30.reshape(1, 197, 448)
        transpose_30 = None
        x_241 = torch._C._nn.linear(
            x_240,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_attn_modules_proj_parameters_bias_,
        )
        x_240 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_attn_modules_proj_parameters_bias_ = (None)
        x_242 = torch.nn.functional.dropout(x_241, 0.0, False, False)
        x_241 = None
        x_243 = x_238 + x_242
        x_238 = x_242 = None
        layer_norm_53 = torch.nn.functional.layer_norm(
            x_243,
            (448,),
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_norm2_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_norm2_parameters_bias_ = (None)
        x_244 = torch._C._nn.linear(
            layer_norm_53,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_53 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_245 = torch._C._nn.gelu(x_244, approximate="none")
        x_244 = None
        x_246 = torch.nn.functional.dropout(x_245, 0.0, False, False)
        x_245 = None
        x_247 = torch._C._nn.linear(
            x_246,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_246 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_248 = torch.nn.functional.dropout(x_247, 0.0, False, False)
        x_247 = None
        x_249 = x_243 + x_248
        x_243 = x_248 = None
        getitem_103 = x_183[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
        input_25 = torch.nn.functional.layer_norm(
            getitem_103,
            (224,),
            l_self_modules_blocks_modules_2_modules_projs_modules_0_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_projs_modules_0_modules_0_parameters_bias_,
            1e-06,
        )
        getitem_103 = l_self_modules_blocks_modules_2_modules_projs_modules_0_modules_0_parameters_weight_ = l_self_modules_blocks_modules_2_modules_projs_modules_0_modules_0_parameters_bias_ = (None)
        input_26 = torch._C._nn.gelu(input_25, approximate="none")
        input_25 = None
        input_27 = torch._C._nn.linear(
            input_26,
            l_self_modules_blocks_modules_2_modules_projs_modules_0_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_projs_modules_0_modules_2_parameters_bias_,
        )
        input_26 = l_self_modules_blocks_modules_2_modules_projs_modules_0_modules_2_parameters_weight_ = l_self_modules_blocks_modules_2_modules_projs_modules_0_modules_2_parameters_bias_ = (None)
        getitem_104 = x_249[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
        input_28 = torch.nn.functional.layer_norm(
            getitem_104,
            (448,),
            l_self_modules_blocks_modules_2_modules_projs_modules_1_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_projs_modules_1_modules_0_parameters_bias_,
            1e-06,
        )
        getitem_104 = l_self_modules_blocks_modules_2_modules_projs_modules_1_modules_0_parameters_weight_ = l_self_modules_blocks_modules_2_modules_projs_modules_1_modules_0_parameters_bias_ = (None)
        input_29 = torch._C._nn.gelu(input_28, approximate="none")
        input_28 = None
        input_30 = torch._C._nn.linear(
            input_29,
            l_self_modules_blocks_modules_2_modules_projs_modules_1_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_projs_modules_1_modules_2_parameters_bias_,
        )
        input_29 = l_self_modules_blocks_modules_2_modules_projs_modules_1_modules_2_parameters_weight_ = l_self_modules_blocks_modules_2_modules_projs_modules_1_modules_2_parameters_bias_ = (None)
        getitem_105 = x_249[(slice(None, None, None), slice(1, None, None), Ellipsis)]
        tmp_8 = torch.cat((input_27, getitem_105), dim=1)
        input_27 = getitem_105 = None
        getitem_106 = tmp_8[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
        layer_norm_56 = torch.nn.functional.layer_norm(
            tmp_8,
            (448,),
            l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_norm1_parameters_bias_,
            1e-06,
        )
        tmp_8 = l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_norm1_parameters_bias_ = (None)
        getitem_107 = layer_norm_56[
            (slice(None, None, None), slice(0, 1, None), Ellipsis)
        ]
        linear_110 = torch._C._nn.linear(
            getitem_107,
            l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wq_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wq_parameters_bias_,
        )
        getitem_107 = l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wq_parameters_weight_ = l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wq_parameters_bias_ = (None)
        reshape_58 = linear_110.reshape(1, 1, 7, 64)
        linear_110 = None
        q_25 = reshape_58.permute(0, 2, 1, 3)
        reshape_58 = None
        linear_111 = torch._C._nn.linear(
            layer_norm_56,
            l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wk_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wk_parameters_bias_,
        )
        l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wk_parameters_weight_ = l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wk_parameters_bias_ = (None)
        reshape_59 = linear_111.reshape(1, 197, 7, 64)
        linear_111 = None
        k_25 = reshape_59.permute(0, 2, 1, 3)
        reshape_59 = None
        linear_112 = torch._C._nn.linear(
            layer_norm_56,
            l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wv_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wv_parameters_bias_,
        )
        layer_norm_56 = l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wv_parameters_weight_ = l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wv_parameters_bias_ = (None)
        reshape_60 = linear_112.reshape(1, 197, 7, 64)
        linear_112 = None
        v_25 = reshape_60.permute(0, 2, 1, 3)
        reshape_60 = None
        transpose_31 = k_25.transpose(-2, -1)
        k_25 = None
        matmul_8 = q_25 @ transpose_31
        q_25 = transpose_31 = None
        attn_12 = matmul_8 * 0.125
        matmul_8 = None
        attn_13 = attn_12.softmax(dim=-1)
        attn_12 = None
        attn_14 = torch.nn.functional.dropout(attn_13, 0.0, False, False)
        attn_13 = None
        matmul_9 = attn_14 @ v_25
        attn_14 = v_25 = None
        transpose_32 = matmul_9.transpose(1, 2)
        matmul_9 = None
        x_250 = transpose_32.reshape(1, 1, 448)
        transpose_32 = None
        x_251 = torch._C._nn.linear(
            x_250,
            l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_250 = l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_252 = torch.nn.functional.dropout(x_251, 0.0, False, False)
        x_251 = None
        x_253 = getitem_106 + x_252
        getitem_106 = x_252 = None
        getitem_108 = x_253[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
        x_253 = None
        input_31 = torch.nn.functional.layer_norm(
            getitem_108,
            (448,),
            l_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_0_parameters_bias_,
            1e-06,
        )
        getitem_108 = l_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_0_parameters_weight_ = l_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_0_parameters_bias_ = (None)
        input_32 = torch._C._nn.gelu(input_31, approximate="none")
        input_31 = None
        input_33 = torch._C._nn.linear(
            input_32,
            l_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_2_parameters_bias_,
        )
        input_32 = l_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_2_parameters_weight_ = l_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_2_parameters_bias_ = (None)
        getitem_109 = x_183[(slice(None, None, None), slice(1, None, None), Ellipsis)]
        tmp_9 = torch.cat((input_33, getitem_109), dim=1)
        input_33 = getitem_109 = None
        getitem_110 = x_183[(slice(None, None, None), slice(1, None, None), Ellipsis)]
        x_183 = None
        tmp_10 = torch.cat((input_30, getitem_110), dim=1)
        input_30 = getitem_110 = None
        getitem_111 = tmp_10[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
        layer_norm_58 = torch.nn.functional.layer_norm(
            tmp_10,
            (224,),
            l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        tmp_10 = l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_norm1_parameters_bias_ = (None)
        getitem_112 = layer_norm_58[
            (slice(None, None, None), slice(0, 1, None), Ellipsis)
        ]
        linear_115 = torch._C._nn.linear(
            getitem_112,
            l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wq_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wq_parameters_bias_,
        )
        getitem_112 = l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wq_parameters_weight_ = l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wq_parameters_bias_ = (None)
        reshape_62 = linear_115.reshape(1, 1, 7, 32)
        linear_115 = None
        q_26 = reshape_62.permute(0, 2, 1, 3)
        reshape_62 = None
        linear_116 = torch._C._nn.linear(
            layer_norm_58,
            l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wk_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wk_parameters_bias_,
        )
        l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wk_parameters_weight_ = l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wk_parameters_bias_ = (None)
        reshape_63 = linear_116.reshape(1, 401, 7, 32)
        linear_116 = None
        k_26 = reshape_63.permute(0, 2, 1, 3)
        reshape_63 = None
        linear_117 = torch._C._nn.linear(
            layer_norm_58,
            l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wv_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wv_parameters_bias_,
        )
        layer_norm_58 = l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wv_parameters_weight_ = l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wv_parameters_bias_ = (None)
        reshape_64 = linear_117.reshape(1, 401, 7, 32)
        linear_117 = None
        v_26 = reshape_64.permute(0, 2, 1, 3)
        reshape_64 = None
        transpose_33 = k_26.transpose(-2, -1)
        k_26 = None
        matmul_10 = q_26 @ transpose_33
        q_26 = transpose_33 = None
        attn_15 = matmul_10 * 0.1767766952966369
        matmul_10 = None
        attn_16 = attn_15.softmax(dim=-1)
        attn_15 = None
        attn_17 = torch.nn.functional.dropout(attn_16, 0.0, False, False)
        attn_16 = None
        matmul_11 = attn_17 @ v_26
        attn_17 = v_26 = None
        transpose_34 = matmul_11.transpose(1, 2)
        matmul_11 = None
        x_254 = transpose_34.reshape(1, 1, 224)
        transpose_34 = None
        x_255 = torch._C._nn.linear(
            x_254,
            l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_254 = l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_256 = torch.nn.functional.dropout(x_255, 0.0, False, False)
        x_255 = None
        x_257 = getitem_111 + x_256
        getitem_111 = x_256 = None
        getitem_113 = x_257[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
        x_257 = None
        input_34 = torch.nn.functional.layer_norm(
            getitem_113,
            (224,),
            l_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_0_parameters_bias_,
            1e-06,
        )
        getitem_113 = l_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_0_parameters_weight_ = l_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_0_parameters_bias_ = (None)
        input_35 = torch._C._nn.gelu(input_34, approximate="none")
        input_34 = None
        input_36 = torch._C._nn.linear(
            input_35,
            l_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_2_parameters_bias_,
        )
        input_35 = l_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_2_parameters_weight_ = l_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_2_parameters_bias_ = (None)
        getitem_114 = x_249[(slice(None, None, None), slice(1, None, None), Ellipsis)]
        x_249 = None
        tmp_11 = torch.cat((input_36, getitem_114), dim=1)
        input_36 = getitem_114 = None
        x_258 = torch.nn.functional.layer_norm(
            tmp_9,
            (224,),
            l_self_modules_norm_modules_0_parameters_weight_,
            l_self_modules_norm_modules_0_parameters_bias_,
            1e-06,
        )
        tmp_9 = (
            l_self_modules_norm_modules_0_parameters_weight_
        ) = l_self_modules_norm_modules_0_parameters_bias_ = None
        x_259 = torch.nn.functional.layer_norm(
            tmp_11,
            (448,),
            l_self_modules_norm_modules_1_parameters_weight_,
            l_self_modules_norm_modules_1_parameters_bias_,
            1e-06,
        )
        tmp_11 = (
            l_self_modules_norm_modules_1_parameters_weight_
        ) = l_self_modules_norm_modules_1_parameters_bias_ = None
        x_260 = x_258[(slice(None, None, None), 0)]
        x_258 = None
        x_261 = x_259[(slice(None, None, None), 0)]
        x_259 = None
        dropout_77 = torch.nn.functional.dropout(x_260, 0.0, False, False)
        x_260 = None
        dropout_78 = torch.nn.functional.dropout(x_261, 0.0, False, False)
        x_261 = None
        linear_120 = torch._C._nn.linear(
            dropout_77,
            l_self_modules_head_modules_0_parameters_weight_,
            l_self_modules_head_modules_0_parameters_bias_,
        )
        dropout_77 = (
            l_self_modules_head_modules_0_parameters_weight_
        ) = l_self_modules_head_modules_0_parameters_bias_ = None
        linear_121 = torch._C._nn.linear(
            dropout_78,
            l_self_modules_head_modules_1_parameters_weight_,
            l_self_modules_head_modules_1_parameters_bias_,
        )
        dropout_78 = (
            l_self_modules_head_modules_1_parameters_weight_
        ) = l_self_modules_head_modules_1_parameters_bias_ = None
        stack = torch.stack([linear_120, linear_121], dim=0)
        linear_120 = linear_121 = None
        x_262 = torch.mean(stack, dim=0)
        stack = None
        return (x_262,)
