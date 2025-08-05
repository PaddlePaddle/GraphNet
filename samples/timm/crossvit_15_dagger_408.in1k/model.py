import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_patch_embed_modules_0_modules_proj_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed_modules_0_modules_proj_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed_modules_0_modules_proj_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed_modules_0_modules_proj_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed_modules_0_modules_proj_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed_modules_0_modules_proj_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_parameters_cls_token_0_: torch.nn.parameter.Parameter,
        L_self_parameters_pos_embed_0_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed_modules_1_modules_proj_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed_modules_1_modules_proj_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed_modules_1_modules_proj_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed_modules_1_modules_proj_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed_modules_1_modules_proj_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed_modules_1_modules_proj_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_patch_embed_modules_0_modules_proj_modules_0_parameters_weight_ = L_self_modules_patch_embed_modules_0_modules_proj_modules_0_parameters_weight_
        l_self_modules_patch_embed_modules_0_modules_proj_modules_0_parameters_bias_ = (
            L_self_modules_patch_embed_modules_0_modules_proj_modules_0_parameters_bias_
        )
        l_self_modules_patch_embed_modules_0_modules_proj_modules_2_parameters_weight_ = L_self_modules_patch_embed_modules_0_modules_proj_modules_2_parameters_weight_
        l_self_modules_patch_embed_modules_0_modules_proj_modules_2_parameters_bias_ = (
            L_self_modules_patch_embed_modules_0_modules_proj_modules_2_parameters_bias_
        )
        l_self_modules_patch_embed_modules_0_modules_proj_modules_4_parameters_weight_ = L_self_modules_patch_embed_modules_0_modules_proj_modules_4_parameters_weight_
        l_self_modules_patch_embed_modules_0_modules_proj_modules_4_parameters_bias_ = (
            L_self_modules_patch_embed_modules_0_modules_proj_modules_4_parameters_bias_
        )
        l_self_parameters_cls_token_0_ = L_self_parameters_cls_token_0_
        l_self_parameters_pos_embed_0_ = L_self_parameters_pos_embed_0_
        l_self_modules_patch_embed_modules_1_modules_proj_modules_0_parameters_weight_ = L_self_modules_patch_embed_modules_1_modules_proj_modules_0_parameters_weight_
        l_self_modules_patch_embed_modules_1_modules_proj_modules_0_parameters_bias_ = (
            L_self_modules_patch_embed_modules_1_modules_proj_modules_0_parameters_bias_
        )
        l_self_modules_patch_embed_modules_1_modules_proj_modules_2_parameters_weight_ = L_self_modules_patch_embed_modules_1_modules_proj_modules_2_parameters_weight_
        l_self_modules_patch_embed_modules_1_modules_proj_modules_2_parameters_bias_ = (
            L_self_modules_patch_embed_modules_1_modules_proj_modules_2_parameters_bias_
        )
        l_self_modules_patch_embed_modules_1_modules_proj_modules_4_parameters_weight_ = L_self_modules_patch_embed_modules_1_modules_proj_modules_4_parameters_weight_
        l_self_modules_patch_embed_modules_1_modules_proj_modules_4_parameters_bias_ = (
            L_self_modules_patch_embed_modules_1_modules_proj_modules_4_parameters_bias_
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
            l_x_, size=(408, 408), mode="bicubic", align_corners=False
        )
        input_1 = torch.conv2d(
            x,
            l_self_modules_patch_embed_modules_0_modules_proj_modules_0_parameters_weight_,
            l_self_modules_patch_embed_modules_0_modules_proj_modules_0_parameters_bias_,
            (4, 4),
            (3, 3),
            (1, 1),
            1,
        )
        x = l_self_modules_patch_embed_modules_0_modules_proj_modules_0_parameters_weight_ = (
            l_self_modules_patch_embed_modules_0_modules_proj_modules_0_parameters_bias_
        ) = None
        input_2 = torch.nn.functional.relu(input_1, inplace=True)
        input_1 = None
        input_3 = torch.conv2d(
            input_2,
            l_self_modules_patch_embed_modules_0_modules_proj_modules_2_parameters_weight_,
            l_self_modules_patch_embed_modules_0_modules_proj_modules_2_parameters_bias_,
            (3, 3),
            (0, 0),
            (1, 1),
            1,
        )
        input_2 = l_self_modules_patch_embed_modules_0_modules_proj_modules_2_parameters_weight_ = (
            l_self_modules_patch_embed_modules_0_modules_proj_modules_2_parameters_bias_
        ) = None
        input_4 = torch.nn.functional.relu(input_3, inplace=True)
        input_3 = None
        input_5 = torch.conv2d(
            input_4,
            l_self_modules_patch_embed_modules_0_modules_proj_modules_4_parameters_weight_,
            l_self_modules_patch_embed_modules_0_modules_proj_modules_4_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_4 = l_self_modules_patch_embed_modules_0_modules_proj_modules_4_parameters_weight_ = (
            l_self_modules_patch_embed_modules_0_modules_proj_modules_4_parameters_bias_
        ) = None
        flatten = input_5.flatten(2)
        input_5 = None
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
        x_2 = torch.nn.functional.interpolate(
            l_x_, size=(384, 384), mode="bicubic", align_corners=False
        )
        l_x_ = None
        input_6 = torch.conv2d(
            x_2,
            l_self_modules_patch_embed_modules_1_modules_proj_modules_0_parameters_weight_,
            l_self_modules_patch_embed_modules_1_modules_proj_modules_0_parameters_bias_,
            (4, 4),
            (3, 3),
            (1, 1),
            1,
        )
        x_2 = l_self_modules_patch_embed_modules_1_modules_proj_modules_0_parameters_weight_ = (
            l_self_modules_patch_embed_modules_1_modules_proj_modules_0_parameters_bias_
        ) = None
        input_7 = torch.nn.functional.relu(input_6, inplace=True)
        input_6 = None
        input_8 = torch.conv2d(
            input_7,
            l_self_modules_patch_embed_modules_1_modules_proj_modules_2_parameters_weight_,
            l_self_modules_patch_embed_modules_1_modules_proj_modules_2_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        input_7 = l_self_modules_patch_embed_modules_1_modules_proj_modules_2_parameters_weight_ = (
            l_self_modules_patch_embed_modules_1_modules_proj_modules_2_parameters_bias_
        ) = None
        input_9 = torch.nn.functional.relu(input_8, inplace=True)
        input_8 = None
        input_10 = torch.conv2d(
            input_9,
            l_self_modules_patch_embed_modules_1_modules_proj_modules_4_parameters_weight_,
            l_self_modules_patch_embed_modules_1_modules_proj_modules_4_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        input_9 = l_self_modules_patch_embed_modules_1_modules_proj_modules_4_parameters_weight_ = (
            l_self_modules_patch_embed_modules_1_modules_proj_modules_4_parameters_bias_
        ) = None
        flatten_1 = input_10.flatten(2)
        input_10 = None
        x_3 = flatten_1.transpose(1, 2)
        flatten_1 = None
        cls_tokens_1 = l_self_parameters_cls_token_1_.expand(1, -1, -1)
        l_self_parameters_cls_token_1_ = None
        x__3 = torch.cat((cls_tokens_1, x_3), dim=1)
        cls_tokens_1 = x_3 = None
        x__4 = x__3 + l_self_parameters_pos_embed_1_
        x__3 = l_self_parameters_pos_embed_1_ = None
        x__5 = torch.nn.functional.dropout(x__4, 0.0, False, False)
        x__4 = None
        layer_norm = torch.nn.functional.layer_norm(
            x__2,
            (192,),
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
        reshape = linear.reshape(1, 1157, 3, 6, 32)
        linear = None
        qkv = reshape.permute(2, 0, 3, 1, 4)
        reshape = None
        unbind = qkv.unbind(0)
        qkv = None
        q = unbind[0]
        k = unbind[1]
        v = unbind[2]
        unbind = None
        x_4 = torch._C._nn.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0
        )
        q = k = v = None
        transpose_2 = x_4.transpose(1, 2)
        x_4 = None
        x_5 = transpose_2.reshape(1, 1157, 192)
        transpose_2 = None
        x_6 = torch._C._nn.linear(
            x_5,
            l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_5 = l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_7 = torch.nn.functional.dropout(x_6, 0.0, False, False)
        x_6 = None
        x_8 = x__2 + x_7
        x__2 = x_7 = None
        layer_norm_1 = torch.nn.functional.layer_norm(
            x_8,
            (192,),
            l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_ = (None)
        x_9 = torch._C._nn.linear(
            layer_norm_1,
            l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_1 = l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_10 = torch._C._nn.gelu(x_9, approximate="none")
        x_9 = None
        x_11 = torch.nn.functional.dropout(x_10, 0.0, False, False)
        x_10 = None
        x_12 = torch._C._nn.linear(
            x_11,
            l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_11 = l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_13 = torch.nn.functional.dropout(x_12, 0.0, False, False)
        x_12 = None
        x_14 = x_8 + x_13
        x_8 = x_13 = None
        layer_norm_2 = torch.nn.functional.layer_norm(
            x__5,
            (384,),
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
        reshape_2 = linear_4.reshape(1, 577, 3, 6, 64)
        linear_4 = None
        qkv_1 = reshape_2.permute(2, 0, 3, 1, 4)
        reshape_2 = None
        unbind_1 = qkv_1.unbind(0)
        qkv_1 = None
        q_1 = unbind_1[0]
        k_1 = unbind_1[1]
        v_1 = unbind_1[2]
        unbind_1 = None
        x_15 = torch._C._nn.scaled_dot_product_attention(
            q_1, k_1, v_1, attn_mask=None, dropout_p=0.0
        )
        q_1 = k_1 = v_1 = None
        transpose_3 = x_15.transpose(1, 2)
        x_15 = None
        x_16 = transpose_3.reshape(1, 577, 384)
        transpose_3 = None
        x_17 = torch._C._nn.linear(
            x_16,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_16 = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_18 = torch.nn.functional.dropout(x_17, 0.0, False, False)
        x_17 = None
        x_19 = x__5 + x_18
        x__5 = x_18 = None
        layer_norm_3 = torch.nn.functional.layer_norm(
            x_19,
            (384,),
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_norm2_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_norm2_parameters_bias_ = (None)
        x_20 = torch._C._nn.linear(
            layer_norm_3,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_3 = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_21 = torch._C._nn.gelu(x_20, approximate="none")
        x_20 = None
        x_22 = torch.nn.functional.dropout(x_21, 0.0, False, False)
        x_21 = None
        x_23 = torch._C._nn.linear(
            x_22,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_22 = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_24 = torch.nn.functional.dropout(x_23, 0.0, False, False)
        x_23 = None
        x_25 = x_19 + x_24
        x_19 = x_24 = None
        layer_norm_4 = torch.nn.functional.layer_norm(
            x_25,
            (384,),
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
        reshape_4 = linear_8.reshape(1, 577, 3, 6, 64)
        linear_8 = None
        qkv_2 = reshape_4.permute(2, 0, 3, 1, 4)
        reshape_4 = None
        unbind_2 = qkv_2.unbind(0)
        qkv_2 = None
        q_2 = unbind_2[0]
        k_2 = unbind_2[1]
        v_2 = unbind_2[2]
        unbind_2 = None
        x_26 = torch._C._nn.scaled_dot_product_attention(
            q_2, k_2, v_2, attn_mask=None, dropout_p=0.0
        )
        q_2 = k_2 = v_2 = None
        transpose_4 = x_26.transpose(1, 2)
        x_26 = None
        x_27 = transpose_4.reshape(1, 577, 384)
        transpose_4 = None
        x_28 = torch._C._nn.linear(
            x_27,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_27 = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_29 = torch.nn.functional.dropout(x_28, 0.0, False, False)
        x_28 = None
        x_30 = x_25 + x_29
        x_25 = x_29 = None
        layer_norm_5 = torch.nn.functional.layer_norm(
            x_30,
            (384,),
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_norm2_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_norm2_parameters_bias_ = (None)
        x_31 = torch._C._nn.linear(
            layer_norm_5,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_5 = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_32 = torch._C._nn.gelu(x_31, approximate="none")
        x_31 = None
        x_33 = torch.nn.functional.dropout(x_32, 0.0, False, False)
        x_32 = None
        x_34 = torch._C._nn.linear(
            x_33,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_33 = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_35 = torch.nn.functional.dropout(x_34, 0.0, False, False)
        x_34 = None
        x_36 = x_30 + x_35
        x_30 = x_35 = None
        layer_norm_6 = torch.nn.functional.layer_norm(
            x_36,
            (384,),
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
        reshape_6 = linear_12.reshape(1, 577, 3, 6, 64)
        linear_12 = None
        qkv_3 = reshape_6.permute(2, 0, 3, 1, 4)
        reshape_6 = None
        unbind_3 = qkv_3.unbind(0)
        qkv_3 = None
        q_3 = unbind_3[0]
        k_3 = unbind_3[1]
        v_3 = unbind_3[2]
        unbind_3 = None
        x_37 = torch._C._nn.scaled_dot_product_attention(
            q_3, k_3, v_3, attn_mask=None, dropout_p=0.0
        )
        q_3 = k_3 = v_3 = None
        transpose_5 = x_37.transpose(1, 2)
        x_37 = None
        x_38 = transpose_5.reshape(1, 577, 384)
        transpose_5 = None
        x_39 = torch._C._nn.linear(
            x_38,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_bias_,
        )
        x_38 = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_bias_ = (None)
        x_40 = torch.nn.functional.dropout(x_39, 0.0, False, False)
        x_39 = None
        x_41 = x_36 + x_40
        x_36 = x_40 = None
        layer_norm_7 = torch.nn.functional.layer_norm(
            x_41,
            (384,),
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_norm2_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_norm2_parameters_bias_ = (None)
        x_42 = torch._C._nn.linear(
            layer_norm_7,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_7 = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_43 = torch._C._nn.gelu(x_42, approximate="none")
        x_42 = None
        x_44 = torch.nn.functional.dropout(x_43, 0.0, False, False)
        x_43 = None
        x_45 = torch._C._nn.linear(
            x_44,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_44 = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_46 = torch.nn.functional.dropout(x_45, 0.0, False, False)
        x_45 = None
        x_47 = x_41 + x_46
        x_41 = x_46 = None
        layer_norm_8 = torch.nn.functional.layer_norm(
            x_47,
            (384,),
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
        reshape_8 = linear_16.reshape(1, 577, 3, 6, 64)
        linear_16 = None
        qkv_4 = reshape_8.permute(2, 0, 3, 1, 4)
        reshape_8 = None
        unbind_4 = qkv_4.unbind(0)
        qkv_4 = None
        q_4 = unbind_4[0]
        k_4 = unbind_4[1]
        v_4 = unbind_4[2]
        unbind_4 = None
        x_48 = torch._C._nn.scaled_dot_product_attention(
            q_4, k_4, v_4, attn_mask=None, dropout_p=0.0
        )
        q_4 = k_4 = v_4 = None
        transpose_6 = x_48.transpose(1, 2)
        x_48 = None
        x_49 = transpose_6.reshape(1, 577, 384)
        transpose_6 = None
        x_50 = torch._C._nn.linear(
            x_49,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_bias_,
        )
        x_49 = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_bias_ = (None)
        x_51 = torch.nn.functional.dropout(x_50, 0.0, False, False)
        x_50 = None
        x_52 = x_47 + x_51
        x_47 = x_51 = None
        layer_norm_9 = torch.nn.functional.layer_norm(
            x_52,
            (384,),
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_norm2_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_norm2_parameters_bias_ = (None)
        x_53 = torch._C._nn.linear(
            layer_norm_9,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_9 = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_54 = torch._C._nn.gelu(x_53, approximate="none")
        x_53 = None
        x_55 = torch.nn.functional.dropout(x_54, 0.0, False, False)
        x_54 = None
        x_56 = torch._C._nn.linear(
            x_55,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_55 = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_57 = torch.nn.functional.dropout(x_56, 0.0, False, False)
        x_56 = None
        x_58 = x_52 + x_57
        x_52 = x_57 = None
        layer_norm_10 = torch.nn.functional.layer_norm(
            x_58,
            (384,),
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
        reshape_10 = linear_20.reshape(1, 577, 3, 6, 64)
        linear_20 = None
        qkv_5 = reshape_10.permute(2, 0, 3, 1, 4)
        reshape_10 = None
        unbind_5 = qkv_5.unbind(0)
        qkv_5 = None
        q_5 = unbind_5[0]
        k_5 = unbind_5[1]
        v_5 = unbind_5[2]
        unbind_5 = None
        x_59 = torch._C._nn.scaled_dot_product_attention(
            q_5, k_5, v_5, attn_mask=None, dropout_p=0.0
        )
        q_5 = k_5 = v_5 = None
        transpose_7 = x_59.transpose(1, 2)
        x_59 = None
        x_60 = transpose_7.reshape(1, 577, 384)
        transpose_7 = None
        x_61 = torch._C._nn.linear(
            x_60,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_bias_,
        )
        x_60 = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_bias_ = (None)
        x_62 = torch.nn.functional.dropout(x_61, 0.0, False, False)
        x_61 = None
        x_63 = x_58 + x_62
        x_58 = x_62 = None
        layer_norm_11 = torch.nn.functional.layer_norm(
            x_63,
            (384,),
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_norm2_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_norm2_parameters_bias_ = (None)
        x_64 = torch._C._nn.linear(
            layer_norm_11,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_11 = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_65 = torch._C._nn.gelu(x_64, approximate="none")
        x_64 = None
        x_66 = torch.nn.functional.dropout(x_65, 0.0, False, False)
        x_65 = None
        x_67 = torch._C._nn.linear(
            x_66,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_66 = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_68 = torch.nn.functional.dropout(x_67, 0.0, False, False)
        x_67 = None
        x_69 = x_63 + x_68
        x_63 = x_68 = None
        getitem_34 = x_14[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
        input_11 = torch.nn.functional.layer_norm(
            getitem_34,
            (192,),
            l_self_modules_blocks_modules_0_modules_projs_modules_0_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_projs_modules_0_modules_0_parameters_bias_,
            1e-06,
        )
        getitem_34 = l_self_modules_blocks_modules_0_modules_projs_modules_0_modules_0_parameters_weight_ = l_self_modules_blocks_modules_0_modules_projs_modules_0_modules_0_parameters_bias_ = (None)
        input_12 = torch._C._nn.gelu(input_11, approximate="none")
        input_11 = None
        input_13 = torch._C._nn.linear(
            input_12,
            l_self_modules_blocks_modules_0_modules_projs_modules_0_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_projs_modules_0_modules_2_parameters_bias_,
        )
        input_12 = l_self_modules_blocks_modules_0_modules_projs_modules_0_modules_2_parameters_weight_ = l_self_modules_blocks_modules_0_modules_projs_modules_0_modules_2_parameters_bias_ = (None)
        getitem_35 = x_69[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
        input_14 = torch.nn.functional.layer_norm(
            getitem_35,
            (384,),
            l_self_modules_blocks_modules_0_modules_projs_modules_1_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_projs_modules_1_modules_0_parameters_bias_,
            1e-06,
        )
        getitem_35 = l_self_modules_blocks_modules_0_modules_projs_modules_1_modules_0_parameters_weight_ = l_self_modules_blocks_modules_0_modules_projs_modules_1_modules_0_parameters_bias_ = (None)
        input_15 = torch._C._nn.gelu(input_14, approximate="none")
        input_14 = None
        input_16 = torch._C._nn.linear(
            input_15,
            l_self_modules_blocks_modules_0_modules_projs_modules_1_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_projs_modules_1_modules_2_parameters_bias_,
        )
        input_15 = l_self_modules_blocks_modules_0_modules_projs_modules_1_modules_2_parameters_weight_ = l_self_modules_blocks_modules_0_modules_projs_modules_1_modules_2_parameters_bias_ = (None)
        getitem_36 = x_69[(slice(None, None, None), slice(1, None, None), Ellipsis)]
        tmp = torch.cat((input_13, getitem_36), dim=1)
        input_13 = getitem_36 = None
        getitem_37 = tmp[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
        layer_norm_14 = torch.nn.functional.layer_norm(
            tmp,
            (384,),
            l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_norm1_parameters_bias_,
            1e-06,
        )
        tmp = l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_norm1_parameters_bias_ = (None)
        getitem_38 = layer_norm_14[
            (slice(None, None, None), slice(0, 1, None), Ellipsis)
        ]
        linear_26 = torch._C._nn.linear(
            getitem_38,
            l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wq_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wq_parameters_bias_,
        )
        getitem_38 = l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wq_parameters_weight_ = l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wq_parameters_bias_ = (None)
        reshape_12 = linear_26.reshape(1, 1, 6, 64)
        linear_26 = None
        q_6 = reshape_12.permute(0, 2, 1, 3)
        reshape_12 = None
        linear_27 = torch._C._nn.linear(
            layer_norm_14,
            l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wk_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wk_parameters_bias_,
        )
        l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wk_parameters_weight_ = l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wk_parameters_bias_ = (None)
        reshape_13 = linear_27.reshape(1, 577, 6, 64)
        linear_27 = None
        k_6 = reshape_13.permute(0, 2, 1, 3)
        reshape_13 = None
        linear_28 = torch._C._nn.linear(
            layer_norm_14,
            l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wv_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wv_parameters_bias_,
        )
        layer_norm_14 = l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wv_parameters_weight_ = l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wv_parameters_bias_ = (None)
        reshape_14 = linear_28.reshape(1, 577, 6, 64)
        linear_28 = None
        v_6 = reshape_14.permute(0, 2, 1, 3)
        reshape_14 = None
        transpose_8 = k_6.transpose(-2, -1)
        k_6 = None
        matmul = q_6 @ transpose_8
        q_6 = transpose_8 = None
        attn = matmul * 0.125
        matmul = None
        attn_1 = attn.softmax(dim=-1)
        attn = None
        attn_2 = torch.nn.functional.dropout(attn_1, 0.0, False, False)
        attn_1 = None
        matmul_1 = attn_2 @ v_6
        attn_2 = v_6 = None
        transpose_9 = matmul_1.transpose(1, 2)
        matmul_1 = None
        x_70 = transpose_9.reshape(1, 1, 384)
        transpose_9 = None
        x_71 = torch._C._nn.linear(
            x_70,
            l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_70 = l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_72 = torch.nn.functional.dropout(x_71, 0.0, False, False)
        x_71 = None
        x_73 = getitem_37 + x_72
        getitem_37 = x_72 = None
        getitem_39 = x_73[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
        x_73 = None
        input_17 = torch.nn.functional.layer_norm(
            getitem_39,
            (384,),
            l_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_0_parameters_bias_,
            1e-06,
        )
        getitem_39 = l_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_0_parameters_weight_ = l_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_0_parameters_bias_ = (None)
        input_18 = torch._C._nn.gelu(input_17, approximate="none")
        input_17 = None
        input_19 = torch._C._nn.linear(
            input_18,
            l_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_2_parameters_bias_,
        )
        input_18 = l_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_2_parameters_weight_ = l_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_2_parameters_bias_ = (None)
        getitem_40 = x_14[(slice(None, None, None), slice(1, None, None), Ellipsis)]
        tmp_1 = torch.cat((input_19, getitem_40), dim=1)
        input_19 = getitem_40 = None
        getitem_41 = x_14[(slice(None, None, None), slice(1, None, None), Ellipsis)]
        x_14 = None
        tmp_2 = torch.cat((input_16, getitem_41), dim=1)
        input_16 = getitem_41 = None
        getitem_42 = tmp_2[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
        layer_norm_16 = torch.nn.functional.layer_norm(
            tmp_2,
            (192,),
            l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        tmp_2 = l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_norm1_parameters_bias_ = (None)
        getitem_43 = layer_norm_16[
            (slice(None, None, None), slice(0, 1, None), Ellipsis)
        ]
        linear_31 = torch._C._nn.linear(
            getitem_43,
            l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wq_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wq_parameters_bias_,
        )
        getitem_43 = l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wq_parameters_weight_ = l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wq_parameters_bias_ = (None)
        reshape_16 = linear_31.reshape(1, 1, 6, 32)
        linear_31 = None
        q_7 = reshape_16.permute(0, 2, 1, 3)
        reshape_16 = None
        linear_32 = torch._C._nn.linear(
            layer_norm_16,
            l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wk_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wk_parameters_bias_,
        )
        l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wk_parameters_weight_ = l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wk_parameters_bias_ = (None)
        reshape_17 = linear_32.reshape(1, 1157, 6, 32)
        linear_32 = None
        k_7 = reshape_17.permute(0, 2, 1, 3)
        reshape_17 = None
        linear_33 = torch._C._nn.linear(
            layer_norm_16,
            l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wv_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wv_parameters_bias_,
        )
        layer_norm_16 = l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wv_parameters_weight_ = l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wv_parameters_bias_ = (None)
        reshape_18 = linear_33.reshape(1, 1157, 6, 32)
        linear_33 = None
        v_7 = reshape_18.permute(0, 2, 1, 3)
        reshape_18 = None
        transpose_10 = k_7.transpose(-2, -1)
        k_7 = None
        matmul_2 = q_7 @ transpose_10
        q_7 = transpose_10 = None
        attn_3 = matmul_2 * 0.1767766952966369
        matmul_2 = None
        attn_4 = attn_3.softmax(dim=-1)
        attn_3 = None
        attn_5 = torch.nn.functional.dropout(attn_4, 0.0, False, False)
        attn_4 = None
        matmul_3 = attn_5 @ v_7
        attn_5 = v_7 = None
        transpose_11 = matmul_3.transpose(1, 2)
        matmul_3 = None
        x_74 = transpose_11.reshape(1, 1, 192)
        transpose_11 = None
        x_75 = torch._C._nn.linear(
            x_74,
            l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_74 = l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_76 = torch.nn.functional.dropout(x_75, 0.0, False, False)
        x_75 = None
        x_77 = getitem_42 + x_76
        getitem_42 = x_76 = None
        getitem_44 = x_77[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
        x_77 = None
        input_20 = torch.nn.functional.layer_norm(
            getitem_44,
            (192,),
            l_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_0_parameters_bias_,
            1e-06,
        )
        getitem_44 = l_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_0_parameters_weight_ = l_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_0_parameters_bias_ = (None)
        input_21 = torch._C._nn.gelu(input_20, approximate="none")
        input_20 = None
        input_22 = torch._C._nn.linear(
            input_21,
            l_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_2_parameters_bias_,
        )
        input_21 = l_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_2_parameters_weight_ = l_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_2_parameters_bias_ = (None)
        getitem_45 = x_69[(slice(None, None, None), slice(1, None, None), Ellipsis)]
        x_69 = None
        tmp_3 = torch.cat((input_22, getitem_45), dim=1)
        input_22 = getitem_45 = None
        layer_norm_18 = torch.nn.functional.layer_norm(
            tmp_1,
            (192,),
            l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_ = (None)
        linear_36 = torch._C._nn.linear(
            layer_norm_18,
            l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_18 = l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_20 = linear_36.reshape(1, 1157, 3, 6, 32)
        linear_36 = None
        qkv_6 = reshape_20.permute(2, 0, 3, 1, 4)
        reshape_20 = None
        unbind_6 = qkv_6.unbind(0)
        qkv_6 = None
        q_8 = unbind_6[0]
        k_8 = unbind_6[1]
        v_8 = unbind_6[2]
        unbind_6 = None
        x_78 = torch._C._nn.scaled_dot_product_attention(
            q_8, k_8, v_8, attn_mask=None, dropout_p=0.0
        )
        q_8 = k_8 = v_8 = None
        transpose_12 = x_78.transpose(1, 2)
        x_78 = None
        x_79 = transpose_12.reshape(1, 1157, 192)
        transpose_12 = None
        x_80 = torch._C._nn.linear(
            x_79,
            l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_79 = l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_81 = torch.nn.functional.dropout(x_80, 0.0, False, False)
        x_80 = None
        x_82 = tmp_1 + x_81
        tmp_1 = x_81 = None
        layer_norm_19 = torch.nn.functional.layer_norm(
            x_82,
            (192,),
            l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_ = (None)
        x_83 = torch._C._nn.linear(
            layer_norm_19,
            l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_19 = l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_84 = torch._C._nn.gelu(x_83, approximate="none")
        x_83 = None
        x_85 = torch.nn.functional.dropout(x_84, 0.0, False, False)
        x_84 = None
        x_86 = torch._C._nn.linear(
            x_85,
            l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_85 = l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_87 = torch.nn.functional.dropout(x_86, 0.0, False, False)
        x_86 = None
        x_88 = x_82 + x_87
        x_82 = x_87 = None
        layer_norm_20 = torch.nn.functional.layer_norm(
            tmp_3,
            (384,),
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm1_parameters_bias_ = (None)
        linear_40 = torch._C._nn.linear(
            layer_norm_20,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_20 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_22 = linear_40.reshape(1, 577, 3, 6, 64)
        linear_40 = None
        qkv_7 = reshape_22.permute(2, 0, 3, 1, 4)
        reshape_22 = None
        unbind_7 = qkv_7.unbind(0)
        qkv_7 = None
        q_9 = unbind_7[0]
        k_9 = unbind_7[1]
        v_9 = unbind_7[2]
        unbind_7 = None
        x_89 = torch._C._nn.scaled_dot_product_attention(
            q_9, k_9, v_9, attn_mask=None, dropout_p=0.0
        )
        q_9 = k_9 = v_9 = None
        transpose_13 = x_89.transpose(1, 2)
        x_89 = None
        x_90 = transpose_13.reshape(1, 577, 384)
        transpose_13 = None
        x_91 = torch._C._nn.linear(
            x_90,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_90 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_92 = torch.nn.functional.dropout(x_91, 0.0, False, False)
        x_91 = None
        x_93 = tmp_3 + x_92
        tmp_3 = x_92 = None
        layer_norm_21 = torch.nn.functional.layer_norm(
            x_93,
            (384,),
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm2_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm2_parameters_bias_ = (None)
        x_94 = torch._C._nn.linear(
            layer_norm_21,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_21 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_95 = torch._C._nn.gelu(x_94, approximate="none")
        x_94 = None
        x_96 = torch.nn.functional.dropout(x_95, 0.0, False, False)
        x_95 = None
        x_97 = torch._C._nn.linear(
            x_96,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_96 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_98 = torch.nn.functional.dropout(x_97, 0.0, False, False)
        x_97 = None
        x_99 = x_93 + x_98
        x_93 = x_98 = None
        layer_norm_22 = torch.nn.functional.layer_norm(
            x_99,
            (384,),
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm1_parameters_bias_ = (None)
        linear_44 = torch._C._nn.linear(
            layer_norm_22,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_22 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_24 = linear_44.reshape(1, 577, 3, 6, 64)
        linear_44 = None
        qkv_8 = reshape_24.permute(2, 0, 3, 1, 4)
        reshape_24 = None
        unbind_8 = qkv_8.unbind(0)
        qkv_8 = None
        q_10 = unbind_8[0]
        k_10 = unbind_8[1]
        v_10 = unbind_8[2]
        unbind_8 = None
        x_100 = torch._C._nn.scaled_dot_product_attention(
            q_10, k_10, v_10, attn_mask=None, dropout_p=0.0
        )
        q_10 = k_10 = v_10 = None
        transpose_14 = x_100.transpose(1, 2)
        x_100 = None
        x_101 = transpose_14.reshape(1, 577, 384)
        transpose_14 = None
        x_102 = torch._C._nn.linear(
            x_101,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_101 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_103 = torch.nn.functional.dropout(x_102, 0.0, False, False)
        x_102 = None
        x_104 = x_99 + x_103
        x_99 = x_103 = None
        layer_norm_23 = torch.nn.functional.layer_norm(
            x_104,
            (384,),
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm2_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm2_parameters_bias_ = (None)
        x_105 = torch._C._nn.linear(
            layer_norm_23,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_23 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_106 = torch._C._nn.gelu(x_105, approximate="none")
        x_105 = None
        x_107 = torch.nn.functional.dropout(x_106, 0.0, False, False)
        x_106 = None
        x_108 = torch._C._nn.linear(
            x_107,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_107 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_109 = torch.nn.functional.dropout(x_108, 0.0, False, False)
        x_108 = None
        x_110 = x_104 + x_109
        x_104 = x_109 = None
        layer_norm_24 = torch.nn.functional.layer_norm(
            x_110,
            (384,),
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm1_parameters_bias_ = (None)
        linear_48 = torch._C._nn.linear(
            layer_norm_24,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_24 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_26 = linear_48.reshape(1, 577, 3, 6, 64)
        linear_48 = None
        qkv_9 = reshape_26.permute(2, 0, 3, 1, 4)
        reshape_26 = None
        unbind_9 = qkv_9.unbind(0)
        qkv_9 = None
        q_11 = unbind_9[0]
        k_11 = unbind_9[1]
        v_11 = unbind_9[2]
        unbind_9 = None
        x_111 = torch._C._nn.scaled_dot_product_attention(
            q_11, k_11, v_11, attn_mask=None, dropout_p=0.0
        )
        q_11 = k_11 = v_11 = None
        transpose_15 = x_111.transpose(1, 2)
        x_111 = None
        x_112 = transpose_15.reshape(1, 577, 384)
        transpose_15 = None
        x_113 = torch._C._nn.linear(
            x_112,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_bias_,
        )
        x_112 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_bias_ = (None)
        x_114 = torch.nn.functional.dropout(x_113, 0.0, False, False)
        x_113 = None
        x_115 = x_110 + x_114
        x_110 = x_114 = None
        layer_norm_25 = torch.nn.functional.layer_norm(
            x_115,
            (384,),
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm2_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm2_parameters_bias_ = (None)
        x_116 = torch._C._nn.linear(
            layer_norm_25,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_25 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_117 = torch._C._nn.gelu(x_116, approximate="none")
        x_116 = None
        x_118 = torch.nn.functional.dropout(x_117, 0.0, False, False)
        x_117 = None
        x_119 = torch._C._nn.linear(
            x_118,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_118 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_120 = torch.nn.functional.dropout(x_119, 0.0, False, False)
        x_119 = None
        x_121 = x_115 + x_120
        x_115 = x_120 = None
        layer_norm_26 = torch.nn.functional.layer_norm(
            x_121,
            (384,),
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_norm1_parameters_bias_ = (None)
        linear_52 = torch._C._nn.linear(
            layer_norm_26,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_26 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_28 = linear_52.reshape(1, 577, 3, 6, 64)
        linear_52 = None
        qkv_10 = reshape_28.permute(2, 0, 3, 1, 4)
        reshape_28 = None
        unbind_10 = qkv_10.unbind(0)
        qkv_10 = None
        q_12 = unbind_10[0]
        k_12 = unbind_10[1]
        v_12 = unbind_10[2]
        unbind_10 = None
        x_122 = torch._C._nn.scaled_dot_product_attention(
            q_12, k_12, v_12, attn_mask=None, dropout_p=0.0
        )
        q_12 = k_12 = v_12 = None
        transpose_16 = x_122.transpose(1, 2)
        x_122 = None
        x_123 = transpose_16.reshape(1, 577, 384)
        transpose_16 = None
        x_124 = torch._C._nn.linear(
            x_123,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_bias_,
        )
        x_123 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_bias_ = (None)
        x_125 = torch.nn.functional.dropout(x_124, 0.0, False, False)
        x_124 = None
        x_126 = x_121 + x_125
        x_121 = x_125 = None
        layer_norm_27 = torch.nn.functional.layer_norm(
            x_126,
            (384,),
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_norm2_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_norm2_parameters_bias_ = (None)
        x_127 = torch._C._nn.linear(
            layer_norm_27,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_27 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_128 = torch._C._nn.gelu(x_127, approximate="none")
        x_127 = None
        x_129 = torch.nn.functional.dropout(x_128, 0.0, False, False)
        x_128 = None
        x_130 = torch._C._nn.linear(
            x_129,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_129 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_131 = torch.nn.functional.dropout(x_130, 0.0, False, False)
        x_130 = None
        x_132 = x_126 + x_131
        x_126 = x_131 = None
        layer_norm_28 = torch.nn.functional.layer_norm(
            x_132,
            (384,),
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_norm1_parameters_bias_ = (None)
        linear_56 = torch._C._nn.linear(
            layer_norm_28,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_28 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_30 = linear_56.reshape(1, 577, 3, 6, 64)
        linear_56 = None
        qkv_11 = reshape_30.permute(2, 0, 3, 1, 4)
        reshape_30 = None
        unbind_11 = qkv_11.unbind(0)
        qkv_11 = None
        q_13 = unbind_11[0]
        k_13 = unbind_11[1]
        v_13 = unbind_11[2]
        unbind_11 = None
        x_133 = torch._C._nn.scaled_dot_product_attention(
            q_13, k_13, v_13, attn_mask=None, dropout_p=0.0
        )
        q_13 = k_13 = v_13 = None
        transpose_17 = x_133.transpose(1, 2)
        x_133 = None
        x_134 = transpose_17.reshape(1, 577, 384)
        transpose_17 = None
        x_135 = torch._C._nn.linear(
            x_134,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_bias_,
        )
        x_134 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_bias_ = (None)
        x_136 = torch.nn.functional.dropout(x_135, 0.0, False, False)
        x_135 = None
        x_137 = x_132 + x_136
        x_132 = x_136 = None
        layer_norm_29 = torch.nn.functional.layer_norm(
            x_137,
            (384,),
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_norm2_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_norm2_parameters_bias_ = (None)
        x_138 = torch._C._nn.linear(
            layer_norm_29,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_29 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_139 = torch._C._nn.gelu(x_138, approximate="none")
        x_138 = None
        x_140 = torch.nn.functional.dropout(x_139, 0.0, False, False)
        x_139 = None
        x_141 = torch._C._nn.linear(
            x_140,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_140 = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_142 = torch.nn.functional.dropout(x_141, 0.0, False, False)
        x_141 = None
        x_143 = x_137 + x_142
        x_137 = x_142 = None
        getitem_64 = x_88[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
        input_23 = torch.nn.functional.layer_norm(
            getitem_64,
            (192,),
            l_self_modules_blocks_modules_1_modules_projs_modules_0_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_projs_modules_0_modules_0_parameters_bias_,
            1e-06,
        )
        getitem_64 = l_self_modules_blocks_modules_1_modules_projs_modules_0_modules_0_parameters_weight_ = l_self_modules_blocks_modules_1_modules_projs_modules_0_modules_0_parameters_bias_ = (None)
        input_24 = torch._C._nn.gelu(input_23, approximate="none")
        input_23 = None
        input_25 = torch._C._nn.linear(
            input_24,
            l_self_modules_blocks_modules_1_modules_projs_modules_0_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_projs_modules_0_modules_2_parameters_bias_,
        )
        input_24 = l_self_modules_blocks_modules_1_modules_projs_modules_0_modules_2_parameters_weight_ = l_self_modules_blocks_modules_1_modules_projs_modules_0_modules_2_parameters_bias_ = (None)
        getitem_65 = x_143[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
        input_26 = torch.nn.functional.layer_norm(
            getitem_65,
            (384,),
            l_self_modules_blocks_modules_1_modules_projs_modules_1_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_projs_modules_1_modules_0_parameters_bias_,
            1e-06,
        )
        getitem_65 = l_self_modules_blocks_modules_1_modules_projs_modules_1_modules_0_parameters_weight_ = l_self_modules_blocks_modules_1_modules_projs_modules_1_modules_0_parameters_bias_ = (None)
        input_27 = torch._C._nn.gelu(input_26, approximate="none")
        input_26 = None
        input_28 = torch._C._nn.linear(
            input_27,
            l_self_modules_blocks_modules_1_modules_projs_modules_1_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_projs_modules_1_modules_2_parameters_bias_,
        )
        input_27 = l_self_modules_blocks_modules_1_modules_projs_modules_1_modules_2_parameters_weight_ = l_self_modules_blocks_modules_1_modules_projs_modules_1_modules_2_parameters_bias_ = (None)
        getitem_66 = x_143[(slice(None, None, None), slice(1, None, None), Ellipsis)]
        tmp_4 = torch.cat((input_25, getitem_66), dim=1)
        input_25 = getitem_66 = None
        getitem_67 = tmp_4[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
        layer_norm_32 = torch.nn.functional.layer_norm(
            tmp_4,
            (384,),
            l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_norm1_parameters_bias_,
            1e-06,
        )
        tmp_4 = l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_norm1_parameters_bias_ = (None)
        getitem_68 = layer_norm_32[
            (slice(None, None, None), slice(0, 1, None), Ellipsis)
        ]
        linear_62 = torch._C._nn.linear(
            getitem_68,
            l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wq_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wq_parameters_bias_,
        )
        getitem_68 = l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wq_parameters_weight_ = l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wq_parameters_bias_ = (None)
        reshape_32 = linear_62.reshape(1, 1, 6, 64)
        linear_62 = None
        q_14 = reshape_32.permute(0, 2, 1, 3)
        reshape_32 = None
        linear_63 = torch._C._nn.linear(
            layer_norm_32,
            l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wk_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wk_parameters_bias_,
        )
        l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wk_parameters_weight_ = l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wk_parameters_bias_ = (None)
        reshape_33 = linear_63.reshape(1, 577, 6, 64)
        linear_63 = None
        k_14 = reshape_33.permute(0, 2, 1, 3)
        reshape_33 = None
        linear_64 = torch._C._nn.linear(
            layer_norm_32,
            l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wv_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wv_parameters_bias_,
        )
        layer_norm_32 = l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wv_parameters_weight_ = l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wv_parameters_bias_ = (None)
        reshape_34 = linear_64.reshape(1, 577, 6, 64)
        linear_64 = None
        v_14 = reshape_34.permute(0, 2, 1, 3)
        reshape_34 = None
        transpose_18 = k_14.transpose(-2, -1)
        k_14 = None
        matmul_4 = q_14 @ transpose_18
        q_14 = transpose_18 = None
        attn_6 = matmul_4 * 0.125
        matmul_4 = None
        attn_7 = attn_6.softmax(dim=-1)
        attn_6 = None
        attn_8 = torch.nn.functional.dropout(attn_7, 0.0, False, False)
        attn_7 = None
        matmul_5 = attn_8 @ v_14
        attn_8 = v_14 = None
        transpose_19 = matmul_5.transpose(1, 2)
        matmul_5 = None
        x_144 = transpose_19.reshape(1, 1, 384)
        transpose_19 = None
        x_145 = torch._C._nn.linear(
            x_144,
            l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_144 = l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_146 = torch.nn.functional.dropout(x_145, 0.0, False, False)
        x_145 = None
        x_147 = getitem_67 + x_146
        getitem_67 = x_146 = None
        getitem_69 = x_147[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
        x_147 = None
        input_29 = torch.nn.functional.layer_norm(
            getitem_69,
            (384,),
            l_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_0_parameters_bias_,
            1e-06,
        )
        getitem_69 = l_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_0_parameters_weight_ = l_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_0_parameters_bias_ = (None)
        input_30 = torch._C._nn.gelu(input_29, approximate="none")
        input_29 = None
        input_31 = torch._C._nn.linear(
            input_30,
            l_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_2_parameters_bias_,
        )
        input_30 = l_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_2_parameters_weight_ = l_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_2_parameters_bias_ = (None)
        getitem_70 = x_88[(slice(None, None, None), slice(1, None, None), Ellipsis)]
        tmp_5 = torch.cat((input_31, getitem_70), dim=1)
        input_31 = getitem_70 = None
        getitem_71 = x_88[(slice(None, None, None), slice(1, None, None), Ellipsis)]
        x_88 = None
        tmp_6 = torch.cat((input_28, getitem_71), dim=1)
        input_28 = getitem_71 = None
        getitem_72 = tmp_6[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
        layer_norm_34 = torch.nn.functional.layer_norm(
            tmp_6,
            (192,),
            l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        tmp_6 = l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_norm1_parameters_bias_ = (None)
        getitem_73 = layer_norm_34[
            (slice(None, None, None), slice(0, 1, None), Ellipsis)
        ]
        linear_67 = torch._C._nn.linear(
            getitem_73,
            l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wq_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wq_parameters_bias_,
        )
        getitem_73 = l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wq_parameters_weight_ = l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wq_parameters_bias_ = (None)
        reshape_36 = linear_67.reshape(1, 1, 6, 32)
        linear_67 = None
        q_15 = reshape_36.permute(0, 2, 1, 3)
        reshape_36 = None
        linear_68 = torch._C._nn.linear(
            layer_norm_34,
            l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wk_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wk_parameters_bias_,
        )
        l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wk_parameters_weight_ = l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wk_parameters_bias_ = (None)
        reshape_37 = linear_68.reshape(1, 1157, 6, 32)
        linear_68 = None
        k_15 = reshape_37.permute(0, 2, 1, 3)
        reshape_37 = None
        linear_69 = torch._C._nn.linear(
            layer_norm_34,
            l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wv_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wv_parameters_bias_,
        )
        layer_norm_34 = l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wv_parameters_weight_ = l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wv_parameters_bias_ = (None)
        reshape_38 = linear_69.reshape(1, 1157, 6, 32)
        linear_69 = None
        v_15 = reshape_38.permute(0, 2, 1, 3)
        reshape_38 = None
        transpose_20 = k_15.transpose(-2, -1)
        k_15 = None
        matmul_6 = q_15 @ transpose_20
        q_15 = transpose_20 = None
        attn_9 = matmul_6 * 0.1767766952966369
        matmul_6 = None
        attn_10 = attn_9.softmax(dim=-1)
        attn_9 = None
        attn_11 = torch.nn.functional.dropout(attn_10, 0.0, False, False)
        attn_10 = None
        matmul_7 = attn_11 @ v_15
        attn_11 = v_15 = None
        transpose_21 = matmul_7.transpose(1, 2)
        matmul_7 = None
        x_148 = transpose_21.reshape(1, 1, 192)
        transpose_21 = None
        x_149 = torch._C._nn.linear(
            x_148,
            l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_148 = l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_150 = torch.nn.functional.dropout(x_149, 0.0, False, False)
        x_149 = None
        x_151 = getitem_72 + x_150
        getitem_72 = x_150 = None
        getitem_74 = x_151[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
        x_151 = None
        input_32 = torch.nn.functional.layer_norm(
            getitem_74,
            (192,),
            l_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_0_parameters_bias_,
            1e-06,
        )
        getitem_74 = l_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_0_parameters_weight_ = l_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_0_parameters_bias_ = (None)
        input_33 = torch._C._nn.gelu(input_32, approximate="none")
        input_32 = None
        input_34 = torch._C._nn.linear(
            input_33,
            l_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_2_parameters_bias_,
        )
        input_33 = l_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_2_parameters_weight_ = l_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_2_parameters_bias_ = (None)
        getitem_75 = x_143[(slice(None, None, None), slice(1, None, None), Ellipsis)]
        x_143 = None
        tmp_7 = torch.cat((input_34, getitem_75), dim=1)
        input_34 = getitem_75 = None
        layer_norm_36 = torch.nn.functional.layer_norm(
            tmp_5,
            (192,),
            l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_ = (None)
        linear_72 = torch._C._nn.linear(
            layer_norm_36,
            l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_36 = l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_40 = linear_72.reshape(1, 1157, 3, 6, 32)
        linear_72 = None
        qkv_12 = reshape_40.permute(2, 0, 3, 1, 4)
        reshape_40 = None
        unbind_12 = qkv_12.unbind(0)
        qkv_12 = None
        q_16 = unbind_12[0]
        k_16 = unbind_12[1]
        v_16 = unbind_12[2]
        unbind_12 = None
        x_152 = torch._C._nn.scaled_dot_product_attention(
            q_16, k_16, v_16, attn_mask=None, dropout_p=0.0
        )
        q_16 = k_16 = v_16 = None
        transpose_22 = x_152.transpose(1, 2)
        x_152 = None
        x_153 = transpose_22.reshape(1, 1157, 192)
        transpose_22 = None
        x_154 = torch._C._nn.linear(
            x_153,
            l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_153 = l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_155 = torch.nn.functional.dropout(x_154, 0.0, False, False)
        x_154 = None
        x_156 = tmp_5 + x_155
        tmp_5 = x_155 = None
        layer_norm_37 = torch.nn.functional.layer_norm(
            x_156,
            (192,),
            l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_ = (None)
        x_157 = torch._C._nn.linear(
            layer_norm_37,
            l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_37 = l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_158 = torch._C._nn.gelu(x_157, approximate="none")
        x_157 = None
        x_159 = torch.nn.functional.dropout(x_158, 0.0, False, False)
        x_158 = None
        x_160 = torch._C._nn.linear(
            x_159,
            l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_159 = l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_161 = torch.nn.functional.dropout(x_160, 0.0, False, False)
        x_160 = None
        x_162 = x_156 + x_161
        x_156 = x_161 = None
        layer_norm_38 = torch.nn.functional.layer_norm(
            tmp_7,
            (384,),
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm1_parameters_bias_ = (None)
        linear_76 = torch._C._nn.linear(
            layer_norm_38,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_38 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_42 = linear_76.reshape(1, 577, 3, 6, 64)
        linear_76 = None
        qkv_13 = reshape_42.permute(2, 0, 3, 1, 4)
        reshape_42 = None
        unbind_13 = qkv_13.unbind(0)
        qkv_13 = None
        q_17 = unbind_13[0]
        k_17 = unbind_13[1]
        v_17 = unbind_13[2]
        unbind_13 = None
        x_163 = torch._C._nn.scaled_dot_product_attention(
            q_17, k_17, v_17, attn_mask=None, dropout_p=0.0
        )
        q_17 = k_17 = v_17 = None
        transpose_23 = x_163.transpose(1, 2)
        x_163 = None
        x_164 = transpose_23.reshape(1, 577, 384)
        transpose_23 = None
        x_165 = torch._C._nn.linear(
            x_164,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_164 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_166 = torch.nn.functional.dropout(x_165, 0.0, False, False)
        x_165 = None
        x_167 = tmp_7 + x_166
        tmp_7 = x_166 = None
        layer_norm_39 = torch.nn.functional.layer_norm(
            x_167,
            (384,),
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm2_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm2_parameters_bias_ = (None)
        x_168 = torch._C._nn.linear(
            layer_norm_39,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_39 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_169 = torch._C._nn.gelu(x_168, approximate="none")
        x_168 = None
        x_170 = torch.nn.functional.dropout(x_169, 0.0, False, False)
        x_169 = None
        x_171 = torch._C._nn.linear(
            x_170,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_170 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_172 = torch.nn.functional.dropout(x_171, 0.0, False, False)
        x_171 = None
        x_173 = x_167 + x_172
        x_167 = x_172 = None
        layer_norm_40 = torch.nn.functional.layer_norm(
            x_173,
            (384,),
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm1_parameters_bias_ = (None)
        linear_80 = torch._C._nn.linear(
            layer_norm_40,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_40 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_44 = linear_80.reshape(1, 577, 3, 6, 64)
        linear_80 = None
        qkv_14 = reshape_44.permute(2, 0, 3, 1, 4)
        reshape_44 = None
        unbind_14 = qkv_14.unbind(0)
        qkv_14 = None
        q_18 = unbind_14[0]
        k_18 = unbind_14[1]
        v_18 = unbind_14[2]
        unbind_14 = None
        x_174 = torch._C._nn.scaled_dot_product_attention(
            q_18, k_18, v_18, attn_mask=None, dropout_p=0.0
        )
        q_18 = k_18 = v_18 = None
        transpose_24 = x_174.transpose(1, 2)
        x_174 = None
        x_175 = transpose_24.reshape(1, 577, 384)
        transpose_24 = None
        x_176 = torch._C._nn.linear(
            x_175,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_175 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_177 = torch.nn.functional.dropout(x_176, 0.0, False, False)
        x_176 = None
        x_178 = x_173 + x_177
        x_173 = x_177 = None
        layer_norm_41 = torch.nn.functional.layer_norm(
            x_178,
            (384,),
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm2_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm2_parameters_bias_ = (None)
        x_179 = torch._C._nn.linear(
            layer_norm_41,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_41 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_180 = torch._C._nn.gelu(x_179, approximate="none")
        x_179 = None
        x_181 = torch.nn.functional.dropout(x_180, 0.0, False, False)
        x_180 = None
        x_182 = torch._C._nn.linear(
            x_181,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_181 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_183 = torch.nn.functional.dropout(x_182, 0.0, False, False)
        x_182 = None
        x_184 = x_178 + x_183
        x_178 = x_183 = None
        layer_norm_42 = torch.nn.functional.layer_norm(
            x_184,
            (384,),
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm1_parameters_bias_ = (None)
        linear_84 = torch._C._nn.linear(
            layer_norm_42,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_42 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_46 = linear_84.reshape(1, 577, 3, 6, 64)
        linear_84 = None
        qkv_15 = reshape_46.permute(2, 0, 3, 1, 4)
        reshape_46 = None
        unbind_15 = qkv_15.unbind(0)
        qkv_15 = None
        q_19 = unbind_15[0]
        k_19 = unbind_15[1]
        v_19 = unbind_15[2]
        unbind_15 = None
        x_185 = torch._C._nn.scaled_dot_product_attention(
            q_19, k_19, v_19, attn_mask=None, dropout_p=0.0
        )
        q_19 = k_19 = v_19 = None
        transpose_25 = x_185.transpose(1, 2)
        x_185 = None
        x_186 = transpose_25.reshape(1, 577, 384)
        transpose_25 = None
        x_187 = torch._C._nn.linear(
            x_186,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_bias_,
        )
        x_186 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_bias_ = (None)
        x_188 = torch.nn.functional.dropout(x_187, 0.0, False, False)
        x_187 = None
        x_189 = x_184 + x_188
        x_184 = x_188 = None
        layer_norm_43 = torch.nn.functional.layer_norm(
            x_189,
            (384,),
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm2_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm2_parameters_bias_ = (None)
        x_190 = torch._C._nn.linear(
            layer_norm_43,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_43 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_191 = torch._C._nn.gelu(x_190, approximate="none")
        x_190 = None
        x_192 = torch.nn.functional.dropout(x_191, 0.0, False, False)
        x_191 = None
        x_193 = torch._C._nn.linear(
            x_192,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_192 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_194 = torch.nn.functional.dropout(x_193, 0.0, False, False)
        x_193 = None
        x_195 = x_189 + x_194
        x_189 = x_194 = None
        layer_norm_44 = torch.nn.functional.layer_norm(
            x_195,
            (384,),
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_norm1_parameters_bias_ = (None)
        linear_88 = torch._C._nn.linear(
            layer_norm_44,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_44 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_48 = linear_88.reshape(1, 577, 3, 6, 64)
        linear_88 = None
        qkv_16 = reshape_48.permute(2, 0, 3, 1, 4)
        reshape_48 = None
        unbind_16 = qkv_16.unbind(0)
        qkv_16 = None
        q_20 = unbind_16[0]
        k_20 = unbind_16[1]
        v_20 = unbind_16[2]
        unbind_16 = None
        x_196 = torch._C._nn.scaled_dot_product_attention(
            q_20, k_20, v_20, attn_mask=None, dropout_p=0.0
        )
        q_20 = k_20 = v_20 = None
        transpose_26 = x_196.transpose(1, 2)
        x_196 = None
        x_197 = transpose_26.reshape(1, 577, 384)
        transpose_26 = None
        x_198 = torch._C._nn.linear(
            x_197,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_bias_,
        )
        x_197 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_bias_ = (None)
        x_199 = torch.nn.functional.dropout(x_198, 0.0, False, False)
        x_198 = None
        x_200 = x_195 + x_199
        x_195 = x_199 = None
        layer_norm_45 = torch.nn.functional.layer_norm(
            x_200,
            (384,),
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_norm2_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_norm2_parameters_bias_ = (None)
        x_201 = torch._C._nn.linear(
            layer_norm_45,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_45 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_202 = torch._C._nn.gelu(x_201, approximate="none")
        x_201 = None
        x_203 = torch.nn.functional.dropout(x_202, 0.0, False, False)
        x_202 = None
        x_204 = torch._C._nn.linear(
            x_203,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_203 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_205 = torch.nn.functional.dropout(x_204, 0.0, False, False)
        x_204 = None
        x_206 = x_200 + x_205
        x_200 = x_205 = None
        layer_norm_46 = torch.nn.functional.layer_norm(
            x_206,
            (384,),
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_norm1_parameters_bias_ = (None)
        linear_92 = torch._C._nn.linear(
            layer_norm_46,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_46 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_50 = linear_92.reshape(1, 577, 3, 6, 64)
        linear_92 = None
        qkv_17 = reshape_50.permute(2, 0, 3, 1, 4)
        reshape_50 = None
        unbind_17 = qkv_17.unbind(0)
        qkv_17 = None
        q_21 = unbind_17[0]
        k_21 = unbind_17[1]
        v_21 = unbind_17[2]
        unbind_17 = None
        x_207 = torch._C._nn.scaled_dot_product_attention(
            q_21, k_21, v_21, attn_mask=None, dropout_p=0.0
        )
        q_21 = k_21 = v_21 = None
        transpose_27 = x_207.transpose(1, 2)
        x_207 = None
        x_208 = transpose_27.reshape(1, 577, 384)
        transpose_27 = None
        x_209 = torch._C._nn.linear(
            x_208,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_bias_,
        )
        x_208 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_bias_ = (None)
        x_210 = torch.nn.functional.dropout(x_209, 0.0, False, False)
        x_209 = None
        x_211 = x_206 + x_210
        x_206 = x_210 = None
        layer_norm_47 = torch.nn.functional.layer_norm(
            x_211,
            (384,),
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_norm2_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_norm2_parameters_bias_ = (None)
        x_212 = torch._C._nn.linear(
            layer_norm_47,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_47 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_213 = torch._C._nn.gelu(x_212, approximate="none")
        x_212 = None
        x_214 = torch.nn.functional.dropout(x_213, 0.0, False, False)
        x_213 = None
        x_215 = torch._C._nn.linear(
            x_214,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_214 = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_216 = torch.nn.functional.dropout(x_215, 0.0, False, False)
        x_215 = None
        x_217 = x_211 + x_216
        x_211 = x_216 = None
        getitem_94 = x_162[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
        input_35 = torch.nn.functional.layer_norm(
            getitem_94,
            (192,),
            l_self_modules_blocks_modules_2_modules_projs_modules_0_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_projs_modules_0_modules_0_parameters_bias_,
            1e-06,
        )
        getitem_94 = l_self_modules_blocks_modules_2_modules_projs_modules_0_modules_0_parameters_weight_ = l_self_modules_blocks_modules_2_modules_projs_modules_0_modules_0_parameters_bias_ = (None)
        input_36 = torch._C._nn.gelu(input_35, approximate="none")
        input_35 = None
        input_37 = torch._C._nn.linear(
            input_36,
            l_self_modules_blocks_modules_2_modules_projs_modules_0_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_projs_modules_0_modules_2_parameters_bias_,
        )
        input_36 = l_self_modules_blocks_modules_2_modules_projs_modules_0_modules_2_parameters_weight_ = l_self_modules_blocks_modules_2_modules_projs_modules_0_modules_2_parameters_bias_ = (None)
        getitem_95 = x_217[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
        input_38 = torch.nn.functional.layer_norm(
            getitem_95,
            (384,),
            l_self_modules_blocks_modules_2_modules_projs_modules_1_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_projs_modules_1_modules_0_parameters_bias_,
            1e-06,
        )
        getitem_95 = l_self_modules_blocks_modules_2_modules_projs_modules_1_modules_0_parameters_weight_ = l_self_modules_blocks_modules_2_modules_projs_modules_1_modules_0_parameters_bias_ = (None)
        input_39 = torch._C._nn.gelu(input_38, approximate="none")
        input_38 = None
        input_40 = torch._C._nn.linear(
            input_39,
            l_self_modules_blocks_modules_2_modules_projs_modules_1_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_projs_modules_1_modules_2_parameters_bias_,
        )
        input_39 = l_self_modules_blocks_modules_2_modules_projs_modules_1_modules_2_parameters_weight_ = l_self_modules_blocks_modules_2_modules_projs_modules_1_modules_2_parameters_bias_ = (None)
        getitem_96 = x_217[(slice(None, None, None), slice(1, None, None), Ellipsis)]
        tmp_8 = torch.cat((input_37, getitem_96), dim=1)
        input_37 = getitem_96 = None
        getitem_97 = tmp_8[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
        layer_norm_50 = torch.nn.functional.layer_norm(
            tmp_8,
            (384,),
            l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_norm1_parameters_bias_,
            1e-06,
        )
        tmp_8 = l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_norm1_parameters_bias_ = (None)
        getitem_98 = layer_norm_50[
            (slice(None, None, None), slice(0, 1, None), Ellipsis)
        ]
        linear_98 = torch._C._nn.linear(
            getitem_98,
            l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wq_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wq_parameters_bias_,
        )
        getitem_98 = l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wq_parameters_weight_ = l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wq_parameters_bias_ = (None)
        reshape_52 = linear_98.reshape(1, 1, 6, 64)
        linear_98 = None
        q_22 = reshape_52.permute(0, 2, 1, 3)
        reshape_52 = None
        linear_99 = torch._C._nn.linear(
            layer_norm_50,
            l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wk_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wk_parameters_bias_,
        )
        l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wk_parameters_weight_ = l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wk_parameters_bias_ = (None)
        reshape_53 = linear_99.reshape(1, 577, 6, 64)
        linear_99 = None
        k_22 = reshape_53.permute(0, 2, 1, 3)
        reshape_53 = None
        linear_100 = torch._C._nn.linear(
            layer_norm_50,
            l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wv_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wv_parameters_bias_,
        )
        layer_norm_50 = l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wv_parameters_weight_ = l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wv_parameters_bias_ = (None)
        reshape_54 = linear_100.reshape(1, 577, 6, 64)
        linear_100 = None
        v_22 = reshape_54.permute(0, 2, 1, 3)
        reshape_54 = None
        transpose_28 = k_22.transpose(-2, -1)
        k_22 = None
        matmul_8 = q_22 @ transpose_28
        q_22 = transpose_28 = None
        attn_12 = matmul_8 * 0.125
        matmul_8 = None
        attn_13 = attn_12.softmax(dim=-1)
        attn_12 = None
        attn_14 = torch.nn.functional.dropout(attn_13, 0.0, False, False)
        attn_13 = None
        matmul_9 = attn_14 @ v_22
        attn_14 = v_22 = None
        transpose_29 = matmul_9.transpose(1, 2)
        matmul_9 = None
        x_218 = transpose_29.reshape(1, 1, 384)
        transpose_29 = None
        x_219 = torch._C._nn.linear(
            x_218,
            l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_218 = l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_220 = torch.nn.functional.dropout(x_219, 0.0, False, False)
        x_219 = None
        x_221 = getitem_97 + x_220
        getitem_97 = x_220 = None
        getitem_99 = x_221[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
        x_221 = None
        input_41 = torch.nn.functional.layer_norm(
            getitem_99,
            (384,),
            l_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_0_parameters_bias_,
            1e-06,
        )
        getitem_99 = l_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_0_parameters_weight_ = l_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_0_parameters_bias_ = (None)
        input_42 = torch._C._nn.gelu(input_41, approximate="none")
        input_41 = None
        input_43 = torch._C._nn.linear(
            input_42,
            l_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_2_parameters_bias_,
        )
        input_42 = l_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_2_parameters_weight_ = l_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_2_parameters_bias_ = (None)
        getitem_100 = x_162[(slice(None, None, None), slice(1, None, None), Ellipsis)]
        tmp_9 = torch.cat((input_43, getitem_100), dim=1)
        input_43 = getitem_100 = None
        getitem_101 = x_162[(slice(None, None, None), slice(1, None, None), Ellipsis)]
        x_162 = None
        tmp_10 = torch.cat((input_40, getitem_101), dim=1)
        input_40 = getitem_101 = None
        getitem_102 = tmp_10[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
        layer_norm_52 = torch.nn.functional.layer_norm(
            tmp_10,
            (192,),
            l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        tmp_10 = l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_norm1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_norm1_parameters_bias_ = (None)
        getitem_103 = layer_norm_52[
            (slice(None, None, None), slice(0, 1, None), Ellipsis)
        ]
        linear_103 = torch._C._nn.linear(
            getitem_103,
            l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wq_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wq_parameters_bias_,
        )
        getitem_103 = l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wq_parameters_weight_ = l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wq_parameters_bias_ = (None)
        reshape_56 = linear_103.reshape(1, 1, 6, 32)
        linear_103 = None
        q_23 = reshape_56.permute(0, 2, 1, 3)
        reshape_56 = None
        linear_104 = torch._C._nn.linear(
            layer_norm_52,
            l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wk_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wk_parameters_bias_,
        )
        l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wk_parameters_weight_ = l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wk_parameters_bias_ = (None)
        reshape_57 = linear_104.reshape(1, 1157, 6, 32)
        linear_104 = None
        k_23 = reshape_57.permute(0, 2, 1, 3)
        reshape_57 = None
        linear_105 = torch._C._nn.linear(
            layer_norm_52,
            l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wv_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wv_parameters_bias_,
        )
        layer_norm_52 = l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wv_parameters_weight_ = l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wv_parameters_bias_ = (None)
        reshape_58 = linear_105.reshape(1, 1157, 6, 32)
        linear_105 = None
        v_23 = reshape_58.permute(0, 2, 1, 3)
        reshape_58 = None
        transpose_30 = k_23.transpose(-2, -1)
        k_23 = None
        matmul_10 = q_23 @ transpose_30
        q_23 = transpose_30 = None
        attn_15 = matmul_10 * 0.1767766952966369
        matmul_10 = None
        attn_16 = attn_15.softmax(dim=-1)
        attn_15 = None
        attn_17 = torch.nn.functional.dropout(attn_16, 0.0, False, False)
        attn_16 = None
        matmul_11 = attn_17 @ v_23
        attn_17 = v_23 = None
        transpose_31 = matmul_11.transpose(1, 2)
        matmul_11 = None
        x_222 = transpose_31.reshape(1, 1, 192)
        transpose_31 = None
        x_223 = torch._C._nn.linear(
            x_222,
            l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_222 = l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_224 = torch.nn.functional.dropout(x_223, 0.0, False, False)
        x_223 = None
        x_225 = getitem_102 + x_224
        getitem_102 = x_224 = None
        getitem_104 = x_225[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
        x_225 = None
        input_44 = torch.nn.functional.layer_norm(
            getitem_104,
            (192,),
            l_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_0_parameters_bias_,
            1e-06,
        )
        getitem_104 = l_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_0_parameters_weight_ = l_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_0_parameters_bias_ = (None)
        input_45 = torch._C._nn.gelu(input_44, approximate="none")
        input_44 = None
        input_46 = torch._C._nn.linear(
            input_45,
            l_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_2_parameters_bias_,
        )
        input_45 = l_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_2_parameters_weight_ = l_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_2_parameters_bias_ = (None)
        getitem_105 = x_217[(slice(None, None, None), slice(1, None, None), Ellipsis)]
        x_217 = None
        tmp_11 = torch.cat((input_46, getitem_105), dim=1)
        input_46 = getitem_105 = None
        x_226 = torch.nn.functional.layer_norm(
            tmp_9,
            (192,),
            l_self_modules_norm_modules_0_parameters_weight_,
            l_self_modules_norm_modules_0_parameters_bias_,
            1e-06,
        )
        tmp_9 = (
            l_self_modules_norm_modules_0_parameters_weight_
        ) = l_self_modules_norm_modules_0_parameters_bias_ = None
        x_227 = torch.nn.functional.layer_norm(
            tmp_11,
            (384,),
            l_self_modules_norm_modules_1_parameters_weight_,
            l_self_modules_norm_modules_1_parameters_bias_,
            1e-06,
        )
        tmp_11 = (
            l_self_modules_norm_modules_1_parameters_weight_
        ) = l_self_modules_norm_modules_1_parameters_bias_ = None
        x_228 = x_226[(slice(None, None, None), 0)]
        x_226 = None
        x_229 = x_227[(slice(None, None, None), 0)]
        x_227 = None
        dropout_68 = torch.nn.functional.dropout(x_228, 0.0, False, False)
        x_228 = None
        dropout_69 = torch.nn.functional.dropout(x_229, 0.0, False, False)
        x_229 = None
        linear_108 = torch._C._nn.linear(
            dropout_68,
            l_self_modules_head_modules_0_parameters_weight_,
            l_self_modules_head_modules_0_parameters_bias_,
        )
        dropout_68 = (
            l_self_modules_head_modules_0_parameters_weight_
        ) = l_self_modules_head_modules_0_parameters_bias_ = None
        linear_109 = torch._C._nn.linear(
            dropout_69,
            l_self_modules_head_modules_1_parameters_weight_,
            l_self_modules_head_modules_1_parameters_bias_,
        )
        dropout_69 = (
            l_self_modules_head_modules_1_parameters_weight_
        ) = l_self_modules_head_modules_1_parameters_bias_ = None
        stack = torch.stack([linear_108, linear_109], dim=0)
        linear_108 = linear_109 = None
        x_230 = torch.mean(stack, dim=0)
        stack = None
        return (x_230,)
