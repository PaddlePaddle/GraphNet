import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_x_: torch.Tensor,
        L_self_modules_patch_embed_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_buffers_relative_coordinates_log_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_buffers_attn_mask_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_buffers_relative_coordinates_log_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_reduction_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_buffers_relative_coordinates_log_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_buffers_attn_mask_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_buffers_relative_coordinates_log_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_reduction_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_buffers_relative_coordinates_log_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_buffers_attn_mask_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_buffers_relative_coordinates_log_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_buffers_relative_coordinates_log_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_buffers_attn_mask_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_buffers_relative_coordinates_log_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_buffers_relative_coordinates_log_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_buffers_attn_mask_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_buffers_relative_coordinates_log_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_buffers_relative_coordinates_log_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_buffers_attn_mask_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_buffers_relative_coordinates_log_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_buffers_relative_coordinates_log_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_buffers_attn_mask_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_buffers_relative_coordinates_log_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_buffers_relative_coordinates_log_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_buffers_attn_mask_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_buffers_relative_coordinates_log_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_buffers_relative_coordinates_log_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_buffers_attn_mask_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_buffers_relative_coordinates_log_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_buffers_relative_coordinates_log_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_buffers_attn_mask_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_buffers_relative_coordinates_log_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_buffers_relative_coordinates_log_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_buffers_attn_mask_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_buffers_relative_coordinates_log_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_reduction_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_buffers_relative_coordinates_log_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_buffers_relative_coordinates_log_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_x_ = L_x_
        l_self_modules_patch_embed_modules_proj_parameters_weight_ = (
            L_self_modules_patch_embed_modules_proj_parameters_weight_
        )
        l_self_modules_patch_embed_modules_proj_parameters_bias_ = (
            L_self_modules_patch_embed_modules_proj_parameters_bias_
        )
        l_self_modules_patch_embed_modules_norm_parameters_weight_ = (
            L_self_modules_patch_embed_modules_norm_parameters_weight_
        )
        l_self_modules_patch_embed_modules_norm_parameters_bias_ = (
            L_self_modules_patch_embed_modules_norm_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_parameters_logit_scale_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_parameters_logit_scale_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_buffers_relative_coordinates_log_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_buffers_relative_coordinates_log_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_buffers_attn_mask_ = (
            L_self_modules_stages_modules_0_modules_blocks_modules_1_buffers_attn_mask_
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_parameters_logit_scale_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_parameters_logit_scale_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_buffers_relative_coordinates_log_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_buffers_relative_coordinates_log_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm3_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm3_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm3_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm3_parameters_bias_
        l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_bias_ = L_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_bias_
        l_self_modules_stages_modules_1_modules_downsample_modules_reduction_parameters_weight_ = L_self_modules_stages_modules_1_modules_downsample_modules_reduction_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_parameters_logit_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_parameters_logit_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_buffers_relative_coordinates_log_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_buffers_relative_coordinates_log_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_buffers_attn_mask_ = (
            L_self_modules_stages_modules_1_modules_blocks_modules_1_buffers_attn_mask_
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_parameters_logit_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_parameters_logit_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_buffers_relative_coordinates_log_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_buffers_relative_coordinates_log_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm3_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm3_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm3_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm3_parameters_bias_
        l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_downsample_modules_reduction_parameters_weight_ = L_self_modules_stages_modules_2_modules_downsample_modules_reduction_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_parameters_logit_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_parameters_logit_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_buffers_relative_coordinates_log_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_buffers_relative_coordinates_log_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_buffers_attn_mask_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_1_buffers_attn_mask_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_parameters_logit_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_parameters_logit_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_buffers_relative_coordinates_log_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_buffers_relative_coordinates_log_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_parameters_logit_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_parameters_logit_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_buffers_relative_coordinates_log_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_buffers_relative_coordinates_log_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_buffers_attn_mask_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_3_buffers_attn_mask_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_parameters_logit_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_parameters_logit_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_buffers_relative_coordinates_log_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_buffers_relative_coordinates_log_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_parameters_logit_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_parameters_logit_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_buffers_relative_coordinates_log_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_buffers_relative_coordinates_log_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_buffers_attn_mask_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_5_buffers_attn_mask_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_parameters_logit_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_parameters_logit_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_buffers_relative_coordinates_log_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_buffers_relative_coordinates_log_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_parameters_logit_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_parameters_logit_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_buffers_relative_coordinates_log_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_buffers_relative_coordinates_log_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_buffers_attn_mask_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_7_buffers_attn_mask_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_parameters_logit_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_parameters_logit_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_buffers_relative_coordinates_log_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_buffers_relative_coordinates_log_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_parameters_logit_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_parameters_logit_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_buffers_relative_coordinates_log_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_buffers_relative_coordinates_log_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_buffers_attn_mask_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_9_buffers_attn_mask_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_parameters_logit_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_parameters_logit_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_buffers_relative_coordinates_log_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_buffers_relative_coordinates_log_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_parameters_logit_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_parameters_logit_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_buffers_relative_coordinates_log_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_buffers_relative_coordinates_log_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_buffers_attn_mask_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_11_buffers_attn_mask_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_parameters_logit_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_parameters_logit_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_buffers_relative_coordinates_log_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_buffers_relative_coordinates_log_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_parameters_logit_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_parameters_logit_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_buffers_relative_coordinates_log_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_buffers_relative_coordinates_log_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_buffers_attn_mask_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_13_buffers_attn_mask_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_parameters_logit_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_parameters_logit_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_buffers_relative_coordinates_log_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_buffers_relative_coordinates_log_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_parameters_logit_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_parameters_logit_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_buffers_relative_coordinates_log_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_buffers_relative_coordinates_log_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_buffers_attn_mask_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_15_buffers_attn_mask_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_parameters_logit_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_parameters_logit_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_buffers_relative_coordinates_log_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_buffers_relative_coordinates_log_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_parameters_logit_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_parameters_logit_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_buffers_relative_coordinates_log_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_buffers_relative_coordinates_log_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_buffers_attn_mask_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_17_buffers_attn_mask_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_parameters_logit_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_parameters_logit_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_buffers_relative_coordinates_log_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_buffers_relative_coordinates_log_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm3_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm3_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm3_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm3_parameters_bias_
        l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_ = L_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_
        l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_bias_ = L_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_bias_
        l_self_modules_stages_modules_3_modules_downsample_modules_reduction_parameters_weight_ = L_self_modules_stages_modules_3_modules_downsample_modules_reduction_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_parameters_logit_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_parameters_logit_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_buffers_relative_coordinates_log_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_buffers_relative_coordinates_log_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_parameters_logit_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_parameters_logit_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_buffers_relative_coordinates_log_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_buffers_relative_coordinates_log_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_parameters_bias_
        l_self_modules_head_modules_fc_parameters_weight_ = (
            L_self_modules_head_modules_fc_parameters_weight_
        )
        l_self_modules_head_modules_fc_parameters_bias_ = (
            L_self_modules_head_modules_fc_parameters_bias_
        )
        x = torch.conv2d(
            l_x_,
            l_self_modules_patch_embed_modules_proj_parameters_weight_,
            l_self_modules_patch_embed_modules_proj_parameters_bias_,
            (4, 4),
            (0, 0),
            (1, 1),
            1,
        )
        l_x_ = (
            l_self_modules_patch_embed_modules_proj_parameters_weight_
        ) = l_self_modules_patch_embed_modules_proj_parameters_bias_ = None
        permute = x.permute(0, 2, 3, 1)
        x = None
        layer_norm = torch.nn.functional.layer_norm(
            permute,
            (96,),
            l_self_modules_patch_embed_modules_norm_parameters_weight_,
            l_self_modules_patch_embed_modules_norm_parameters_bias_,
            1e-05,
        )
        permute = (
            l_self_modules_patch_embed_modules_norm_parameters_weight_
        ) = l_self_modules_patch_embed_modules_norm_parameters_bias_ = None
        x_1 = layer_norm.permute(0, 3, 1, 2)
        layer_norm = None
        x_2 = x_1.permute(0, 2, 3, 1)
        x_1 = None
        x_3 = torch._C._nn.pad(x_2, (0, 0, 0, 0, 0, 0), "constant", None)
        x_4 = x_3.view(1, 8, 7, 8, 7, 96)
        x_3 = None
        permute_3 = x_4.permute(0, 1, 3, 2, 4, 5)
        x_4 = None
        contiguous = permute_3.contiguous()
        permute_3 = None
        windows = contiguous.view(-1, 7, 7, 96)
        contiguous = None
        x_windows = windows.view(-1, 49, 96)
        windows = None
        linear = torch._C._nn.linear(
            x_windows,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        x_windows = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_3 = linear.view(64, 49, 3, 3, 32)
        linear = None
        qkv = view_3.permute(2, 0, 3, 1, 4)
        view_3 = None
        unbind = qkv.unbind(0)
        qkv = None
        query = unbind[0]
        key = unbind[1]
        value = unbind[2]
        unbind = None
        normalize = torch.nn.functional.normalize(query, dim=-1)
        query = None
        normalize_1 = torch.nn.functional.normalize(key, dim=-1)
        key = None
        transpose = normalize_1.transpose(-2, -1)
        normalize_1 = None
        attn = normalize @ transpose
        normalize = transpose = None
        reshape = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_parameters_logit_scale_.reshape(
            1, 3, 1, 1
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_parameters_logit_scale_ = (
            None
        )
        clamp = torch.clamp(reshape, max=4.605170185988092)
        reshape = None
        logit_scale = clamp.exp()
        clamp = None
        attn_1 = attn * logit_scale
        attn = logit_scale = None
        x_5 = torch._C._nn.linear(
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_6 = torch.nn.functional.relu(x_5, inplace=False)
        x_5 = None
        x_7 = torch.nn.functional.dropout(x_6, 0.125, False, False)
        x_6 = None
        x_8 = torch._C._nn.linear(
            x_7,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_7 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_9 = torch.nn.functional.dropout(x_8, 0.0, False, False)
        x_8 = None
        transpose_1 = x_9.transpose(1, 0)
        x_9 = None
        relative_position_bias = transpose_1.reshape(3, 49, 49)
        transpose_1 = None
        relative_position_bias_1 = relative_position_bias.unsqueeze(0)
        relative_position_bias = None
        attn_2 = attn_1 + relative_position_bias_1
        attn_1 = relative_position_bias_1 = None
        attn_3 = attn_2.softmax(dim=-1)
        attn_2 = None
        attn_4 = torch.nn.functional.dropout(attn_3, 0.0, False, False)
        attn_3 = None
        matmul_1 = attn_4 @ value
        attn_4 = value = None
        transpose_2 = matmul_1.transpose(1, 2)
        matmul_1 = None
        x_10 = transpose_2.reshape(64, 49, -1)
        transpose_2 = None
        x_11 = torch._C._nn.linear(
            x_10,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_10 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_12 = torch.nn.functional.dropout(x_11, 0.0, False, False)
        x_11 = None
        attn_windows = x_12.view(-1, 7, 7, 96)
        x_12 = None
        x_13 = attn_windows.view(-1, 8, 8, 7, 7, 96)
        attn_windows = None
        permute_5 = x_13.permute(0, 1, 3, 2, 4, 5)
        x_13 = None
        contiguous_1 = permute_5.contiguous()
        permute_5 = None
        x_14 = contiguous_1.view(-1, 56, 56, 96)
        contiguous_1 = None
        getitem_7 = x_14[
            (
                slice(None, None, None),
                slice(None, 56, None),
                slice(None, 56, None),
                slice(None, None, None),
            )
        ]
        x_14 = None
        x_15 = getitem_7.contiguous()
        getitem_7 = None
        layer_norm_1 = torch.nn.functional.layer_norm(
            x_15,
            (96,),
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_15 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (None)
        x_16 = x_2 + layer_norm_1
        x_2 = layer_norm_1 = None
        x_17 = x_16.reshape(1, -1, 96)
        x_16 = None
        x_18 = torch._C._nn.linear(
            x_17,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_19 = torch._C._nn.gelu(x_18, approximate="none")
        x_18 = None
        x_20 = torch.nn.functional.dropout(x_19, 0.0, False, False)
        x_19 = None
        x_21 = torch._C._nn.linear(
            x_20,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_20 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_22 = torch.nn.functional.dropout(x_21, 0.0, False, False)
        x_21 = None
        layer_norm_2 = torch.nn.functional.layer_norm(
            x_22,
            (96,),
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_22 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_23 = x_17 + layer_norm_2
        x_17 = layer_norm_2 = None
        x_24 = x_23.reshape(1, 56, 56, 96)
        x_23 = None
        x_25 = torch.roll(x_24, shifts=(-3, -3), dims=(1, 2))
        x_26 = torch._C._nn.pad(x_25, (0, 0, 0, 0, 0, 0), "constant", None)
        x_25 = None
        x_27 = x_26.view(1, 8, 7, 8, 7, 96)
        x_26 = None
        permute_6 = x_27.permute(0, 1, 3, 2, 4, 5)
        x_27 = None
        contiguous_3 = permute_6.contiguous()
        permute_6 = None
        windows_1 = contiguous_3.view(-1, 7, 7, 96)
        contiguous_3 = None
        x_windows_1 = windows_1.view(-1, 49, 96)
        windows_1 = None
        linear_6 = torch._C._nn.linear(
            x_windows_1,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_,
        )
        x_windows_1 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_10 = linear_6.view(64, 49, 3, 3, 32)
        linear_6 = None
        qkv_1 = view_10.permute(2, 0, 3, 1, 4)
        view_10 = None
        unbind_1 = qkv_1.unbind(0)
        qkv_1 = None
        query_1 = unbind_1[0]
        key_1 = unbind_1[1]
        value_1 = unbind_1[2]
        unbind_1 = None
        normalize_2 = torch.nn.functional.normalize(query_1, dim=-1)
        query_1 = None
        normalize_3 = torch.nn.functional.normalize(key_1, dim=-1)
        key_1 = None
        transpose_3 = normalize_3.transpose(-2, -1)
        normalize_3 = None
        attn_5 = normalize_2 @ transpose_3
        normalize_2 = transpose_3 = None
        reshape_5 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_parameters_logit_scale_.reshape(
            1, 3, 1, 1
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_parameters_logit_scale_ = (
            None
        )
        clamp_1 = torch.clamp(reshape_5, max=4.605170185988092)
        reshape_5 = None
        logit_scale_1 = clamp_1.exp()
        clamp_1 = None
        attn_6 = attn_5 * logit_scale_1
        attn_5 = logit_scale_1 = None
        x_28 = torch._C._nn.linear(
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_29 = torch.nn.functional.relu(x_28, inplace=False)
        x_28 = None
        x_30 = torch.nn.functional.dropout(x_29, 0.125, False, False)
        x_29 = None
        x_31 = torch._C._nn.linear(
            x_30,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_30 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_32 = torch.nn.functional.dropout(x_31, 0.0, False, False)
        x_31 = None
        transpose_4 = x_32.transpose(1, 0)
        x_32 = None
        relative_position_bias_2 = transpose_4.reshape(3, 49, 49)
        transpose_4 = None
        relative_position_bias_3 = relative_position_bias_2.unsqueeze(0)
        relative_position_bias_2 = None
        attn_7 = attn_6 + relative_position_bias_3
        attn_6 = relative_position_bias_3 = None
        attn_8 = attn_7.view(1, 64, 3, 49, 49)
        attn_7 = None
        unsqueeze_2 = l_self_modules_stages_modules_0_modules_blocks_modules_1_buffers_attn_mask_.unsqueeze(
            1
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_buffers_attn_mask_ = (
            None
        )
        unsqueeze_3 = unsqueeze_2.unsqueeze(0)
        unsqueeze_2 = None
        attn_9 = attn_8 + unsqueeze_3
        attn_8 = unsqueeze_3 = None
        attn_10 = attn_9.view(-1, 3, 49, 49)
        attn_9 = None
        attn_11 = attn_10.softmax(dim=-1)
        attn_10 = None
        attn_12 = torch.nn.functional.dropout(attn_11, 0.0, False, False)
        attn_11 = None
        matmul_3 = attn_12 @ value_1
        attn_12 = value_1 = None
        transpose_5 = matmul_3.transpose(1, 2)
        matmul_3 = None
        x_33 = transpose_5.reshape(64, 49, -1)
        transpose_5 = None
        x_34 = torch._C._nn.linear(
            x_33,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_33 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_35 = torch.nn.functional.dropout(x_34, 0.0, False, False)
        x_34 = None
        attn_windows_1 = x_35.view(-1, 7, 7, 96)
        x_35 = None
        x_36 = attn_windows_1.view(-1, 8, 8, 7, 7, 96)
        attn_windows_1 = None
        permute_8 = x_36.permute(0, 1, 3, 2, 4, 5)
        x_36 = None
        contiguous_4 = permute_8.contiguous()
        permute_8 = None
        x_37 = contiguous_4.view(-1, 56, 56, 96)
        contiguous_4 = None
        getitem_11 = x_37[
            (
                slice(None, None, None),
                slice(None, 56, None),
                slice(None, 56, None),
                slice(None, None, None),
            )
        ]
        x_37 = None
        x_38 = getitem_11.contiguous()
        getitem_11 = None
        x_39 = torch.roll(x_38, shifts=(3, 3), dims=(1, 2))
        x_38 = None
        layer_norm_3 = torch.nn.functional.layer_norm(
            x_39,
            (96,),
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_39 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_40 = x_24 + layer_norm_3
        x_24 = layer_norm_3 = None
        x_41 = x_40.reshape(1, -1, 96)
        x_40 = None
        x_42 = torch._C._nn.linear(
            x_41,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_43 = torch._C._nn.gelu(x_42, approximate="none")
        x_42 = None
        x_44 = torch.nn.functional.dropout(x_43, 0.0, False, False)
        x_43 = None
        x_45 = torch._C._nn.linear(
            x_44,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_44 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_46 = torch.nn.functional.dropout(x_45, 0.0, False, False)
        x_45 = None
        layer_norm_4 = torch.nn.functional.layer_norm(
            x_46,
            (96,),
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_46 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_47 = x_41 + layer_norm_4
        x_41 = layer_norm_4 = None
        x_48 = torch.nn.functional.layer_norm(
            x_47,
            (96,),
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm3_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm3_parameters_bias_,
            1e-05,
        )
        x_47 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm3_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm3_parameters_bias_ = (None)
        x_49 = x_48.reshape(1, 56, 56, 96)
        x_48 = None
        x_50 = x_49.permute(0, 3, 1, 2)
        x_49 = None
        x_51 = x_50.permute(0, 2, 3, 1)
        x_50 = None
        x_52 = torch._C._nn.pad(x_51, (0, 0, 0, 0, 0, 0), "constant", None)
        x_51 = None
        reshape_10 = x_52.reshape(1, 28, 2, 28, 2, 96)
        x_52 = None
        permute_11 = reshape_10.permute(0, 1, 3, 4, 2, 5)
        reshape_10 = None
        x_53 = permute_11.flatten(3)
        permute_11 = None
        x_54 = torch.nn.functional.layer_norm(
            x_53,
            (384,),
            l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_bias_,
            1e-05,
        )
        x_53 = l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_bias_ = (None)
        x_55 = torch._C._nn.linear(
            x_54,
            l_self_modules_stages_modules_1_modules_downsample_modules_reduction_parameters_weight_,
            None,
        )
        x_54 = l_self_modules_stages_modules_1_modules_downsample_modules_reduction_parameters_weight_ = (None)
        x_56 = torch._C._nn.pad(x_55, (0, 0, 0, 0, 0, 0), "constant", None)
        x_57 = x_56.view(1, 4, 7, 4, 7, 192)
        x_56 = None
        permute_12 = x_57.permute(0, 1, 3, 2, 4, 5)
        x_57 = None
        contiguous_6 = permute_12.contiguous()
        permute_12 = None
        windows_2 = contiguous_6.view(-1, 7, 7, 192)
        contiguous_6 = None
        x_windows_2 = windows_2.view(-1, 49, 192)
        windows_2 = None
        linear_13 = torch._C._nn.linear(
            x_windows_2,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        x_windows_2 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_19 = linear_13.view(16, 49, 3, 6, 32)
        linear_13 = None
        qkv_2 = view_19.permute(2, 0, 3, 1, 4)
        view_19 = None
        unbind_2 = qkv_2.unbind(0)
        qkv_2 = None
        query_2 = unbind_2[0]
        key_2 = unbind_2[1]
        value_2 = unbind_2[2]
        unbind_2 = None
        normalize_4 = torch.nn.functional.normalize(query_2, dim=-1)
        query_2 = None
        normalize_5 = torch.nn.functional.normalize(key_2, dim=-1)
        key_2 = None
        transpose_6 = normalize_5.transpose(-2, -1)
        normalize_5 = None
        attn_13 = normalize_4 @ transpose_6
        normalize_4 = transpose_6 = None
        reshape_11 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_parameters_logit_scale_.reshape(
            1, 6, 1, 1
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_parameters_logit_scale_ = (
            None
        )
        clamp_2 = torch.clamp(reshape_11, max=4.605170185988092)
        reshape_11 = None
        logit_scale_2 = clamp_2.exp()
        clamp_2 = None
        attn_14 = attn_13 * logit_scale_2
        attn_13 = logit_scale_2 = None
        x_58 = torch._C._nn.linear(
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_59 = torch.nn.functional.relu(x_58, inplace=False)
        x_58 = None
        x_60 = torch.nn.functional.dropout(x_59, 0.125, False, False)
        x_59 = None
        x_61 = torch._C._nn.linear(
            x_60,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_60 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_62 = torch.nn.functional.dropout(x_61, 0.0, False, False)
        x_61 = None
        transpose_7 = x_62.transpose(1, 0)
        x_62 = None
        relative_position_bias_4 = transpose_7.reshape(6, 49, 49)
        transpose_7 = None
        relative_position_bias_5 = relative_position_bias_4.unsqueeze(0)
        relative_position_bias_4 = None
        attn_15 = attn_14 + relative_position_bias_5
        attn_14 = relative_position_bias_5 = None
        attn_16 = attn_15.softmax(dim=-1)
        attn_15 = None
        attn_17 = torch.nn.functional.dropout(attn_16, 0.0, False, False)
        attn_16 = None
        matmul_5 = attn_17 @ value_2
        attn_17 = value_2 = None
        transpose_8 = matmul_5.transpose(1, 2)
        matmul_5 = None
        x_63 = transpose_8.reshape(16, 49, -1)
        transpose_8 = None
        x_64 = torch._C._nn.linear(
            x_63,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_63 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_65 = torch.nn.functional.dropout(x_64, 0.0, False, False)
        x_64 = None
        attn_windows_2 = x_65.view(-1, 7, 7, 192)
        x_65 = None
        x_66 = attn_windows_2.view(-1, 4, 4, 7, 7, 192)
        attn_windows_2 = None
        permute_14 = x_66.permute(0, 1, 3, 2, 4, 5)
        x_66 = None
        contiguous_7 = permute_14.contiguous()
        permute_14 = None
        x_67 = contiguous_7.view(-1, 28, 28, 192)
        contiguous_7 = None
        getitem_15 = x_67[
            (
                slice(None, None, None),
                slice(None, 28, None),
                slice(None, 28, None),
                slice(None, None, None),
            )
        ]
        x_67 = None
        x_68 = getitem_15.contiguous()
        getitem_15 = None
        layer_norm_7 = torch.nn.functional.layer_norm(
            x_68,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_68 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (None)
        x_69 = x_55 + layer_norm_7
        x_55 = layer_norm_7 = None
        x_70 = x_69.reshape(1, -1, 192)
        x_69 = None
        x_71 = torch._C._nn.linear(
            x_70,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_72 = torch._C._nn.gelu(x_71, approximate="none")
        x_71 = None
        x_73 = torch.nn.functional.dropout(x_72, 0.0, False, False)
        x_72 = None
        x_74 = torch._C._nn.linear(
            x_73,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_73 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_75 = torch.nn.functional.dropout(x_74, 0.0, False, False)
        x_74 = None
        layer_norm_8 = torch.nn.functional.layer_norm(
            x_75,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_75 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_76 = x_70 + layer_norm_8
        x_70 = layer_norm_8 = None
        x_77 = x_76.reshape(1, 28, 28, 192)
        x_76 = None
        x_78 = torch.roll(x_77, shifts=(-3, -3), dims=(1, 2))
        x_79 = torch._C._nn.pad(x_78, (0, 0, 0, 0, 0, 0), "constant", None)
        x_78 = None
        x_80 = x_79.view(1, 4, 7, 4, 7, 192)
        x_79 = None
        permute_15 = x_80.permute(0, 1, 3, 2, 4, 5)
        x_80 = None
        contiguous_9 = permute_15.contiguous()
        permute_15 = None
        windows_3 = contiguous_9.view(-1, 7, 7, 192)
        contiguous_9 = None
        x_windows_3 = windows_3.view(-1, 49, 192)
        windows_3 = None
        linear_19 = torch._C._nn.linear(
            x_windows_3,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_,
        )
        x_windows_3 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_26 = linear_19.view(16, 49, 3, 6, 32)
        linear_19 = None
        qkv_3 = view_26.permute(2, 0, 3, 1, 4)
        view_26 = None
        unbind_3 = qkv_3.unbind(0)
        qkv_3 = None
        query_3 = unbind_3[0]
        key_3 = unbind_3[1]
        value_3 = unbind_3[2]
        unbind_3 = None
        normalize_6 = torch.nn.functional.normalize(query_3, dim=-1)
        query_3 = None
        normalize_7 = torch.nn.functional.normalize(key_3, dim=-1)
        key_3 = None
        transpose_9 = normalize_7.transpose(-2, -1)
        normalize_7 = None
        attn_18 = normalize_6 @ transpose_9
        normalize_6 = transpose_9 = None
        reshape_16 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_parameters_logit_scale_.reshape(
            1, 6, 1, 1
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_parameters_logit_scale_ = (
            None
        )
        clamp_3 = torch.clamp(reshape_16, max=4.605170185988092)
        reshape_16 = None
        logit_scale_3 = clamp_3.exp()
        clamp_3 = None
        attn_19 = attn_18 * logit_scale_3
        attn_18 = logit_scale_3 = None
        x_81 = torch._C._nn.linear(
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_82 = torch.nn.functional.relu(x_81, inplace=False)
        x_81 = None
        x_83 = torch.nn.functional.dropout(x_82, 0.125, False, False)
        x_82 = None
        x_84 = torch._C._nn.linear(
            x_83,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_83 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_85 = torch.nn.functional.dropout(x_84, 0.0, False, False)
        x_84 = None
        transpose_10 = x_85.transpose(1, 0)
        x_85 = None
        relative_position_bias_6 = transpose_10.reshape(6, 49, 49)
        transpose_10 = None
        relative_position_bias_7 = relative_position_bias_6.unsqueeze(0)
        relative_position_bias_6 = None
        attn_20 = attn_19 + relative_position_bias_7
        attn_19 = relative_position_bias_7 = None
        attn_21 = attn_20.view(1, 16, 6, 49, 49)
        attn_20 = None
        unsqueeze_6 = l_self_modules_stages_modules_1_modules_blocks_modules_1_buffers_attn_mask_.unsqueeze(
            1
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_buffers_attn_mask_ = (
            None
        )
        unsqueeze_7 = unsqueeze_6.unsqueeze(0)
        unsqueeze_6 = None
        attn_22 = attn_21 + unsqueeze_7
        attn_21 = unsqueeze_7 = None
        attn_23 = attn_22.view(-1, 6, 49, 49)
        attn_22 = None
        attn_24 = attn_23.softmax(dim=-1)
        attn_23 = None
        attn_25 = torch.nn.functional.dropout(attn_24, 0.0, False, False)
        attn_24 = None
        matmul_7 = attn_25 @ value_3
        attn_25 = value_3 = None
        transpose_11 = matmul_7.transpose(1, 2)
        matmul_7 = None
        x_86 = transpose_11.reshape(16, 49, -1)
        transpose_11 = None
        x_87 = torch._C._nn.linear(
            x_86,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_86 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_88 = torch.nn.functional.dropout(x_87, 0.0, False, False)
        x_87 = None
        attn_windows_3 = x_88.view(-1, 7, 7, 192)
        x_88 = None
        x_89 = attn_windows_3.view(-1, 4, 4, 7, 7, 192)
        attn_windows_3 = None
        permute_17 = x_89.permute(0, 1, 3, 2, 4, 5)
        x_89 = None
        contiguous_10 = permute_17.contiguous()
        permute_17 = None
        x_90 = contiguous_10.view(-1, 28, 28, 192)
        contiguous_10 = None
        getitem_19 = x_90[
            (
                slice(None, None, None),
                slice(None, 28, None),
                slice(None, 28, None),
                slice(None, None, None),
            )
        ]
        x_90 = None
        x_91 = getitem_19.contiguous()
        getitem_19 = None
        x_92 = torch.roll(x_91, shifts=(3, 3), dims=(1, 2))
        x_91 = None
        layer_norm_9 = torch.nn.functional.layer_norm(
            x_92,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_92 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_93 = x_77 + layer_norm_9
        x_77 = layer_norm_9 = None
        x_94 = x_93.reshape(1, -1, 192)
        x_93 = None
        x_95 = torch._C._nn.linear(
            x_94,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_96 = torch._C._nn.gelu(x_95, approximate="none")
        x_95 = None
        x_97 = torch.nn.functional.dropout(x_96, 0.0, False, False)
        x_96 = None
        x_98 = torch._C._nn.linear(
            x_97,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_97 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_99 = torch.nn.functional.dropout(x_98, 0.0, False, False)
        x_98 = None
        layer_norm_10 = torch.nn.functional.layer_norm(
            x_99,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_99 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_100 = x_94 + layer_norm_10
        x_94 = layer_norm_10 = None
        x_101 = torch.nn.functional.layer_norm(
            x_100,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm3_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm3_parameters_bias_,
            1e-05,
        )
        x_100 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm3_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm3_parameters_bias_ = (None)
        x_102 = x_101.reshape(1, 28, 28, 192)
        x_101 = None
        x_103 = x_102.permute(0, 3, 1, 2)
        x_102 = None
        x_104 = x_103.permute(0, 2, 3, 1)
        x_103 = None
        x_105 = torch._C._nn.pad(x_104, (0, 0, 0, 0, 0, 0), "constant", None)
        x_104 = None
        reshape_21 = x_105.reshape(1, 14, 2, 14, 2, 192)
        x_105 = None
        permute_20 = reshape_21.permute(0, 1, 3, 4, 2, 5)
        reshape_21 = None
        x_106 = permute_20.flatten(3)
        permute_20 = None
        x_107 = torch.nn.functional.layer_norm(
            x_106,
            (768,),
            l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_bias_,
            1e-05,
        )
        x_106 = l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_bias_ = (None)
        x_108 = torch._C._nn.linear(
            x_107,
            l_self_modules_stages_modules_2_modules_downsample_modules_reduction_parameters_weight_,
            None,
        )
        x_107 = l_self_modules_stages_modules_2_modules_downsample_modules_reduction_parameters_weight_ = (None)
        x_109 = torch._C._nn.pad(x_108, (0, 0, 0, 0, 0, 0), "constant", None)
        x_110 = x_109.view(1, 2, 7, 2, 7, 384)
        x_109 = None
        permute_21 = x_110.permute(0, 1, 3, 2, 4, 5)
        x_110 = None
        contiguous_12 = permute_21.contiguous()
        permute_21 = None
        windows_4 = contiguous_12.view(-1, 7, 7, 384)
        contiguous_12 = None
        x_windows_4 = windows_4.view(-1, 49, 384)
        windows_4 = None
        linear_26 = torch._C._nn.linear(
            x_windows_4,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        x_windows_4 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_35 = linear_26.view(4, 49, 3, 12, 32)
        linear_26 = None
        qkv_4 = view_35.permute(2, 0, 3, 1, 4)
        view_35 = None
        unbind_4 = qkv_4.unbind(0)
        qkv_4 = None
        query_4 = unbind_4[0]
        key_4 = unbind_4[1]
        value_4 = unbind_4[2]
        unbind_4 = None
        normalize_8 = torch.nn.functional.normalize(query_4, dim=-1)
        query_4 = None
        normalize_9 = torch.nn.functional.normalize(key_4, dim=-1)
        key_4 = None
        transpose_12 = normalize_9.transpose(-2, -1)
        normalize_9 = None
        attn_26 = normalize_8 @ transpose_12
        normalize_8 = transpose_12 = None
        reshape_22 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_parameters_logit_scale_.reshape(
            1, 12, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_parameters_logit_scale_ = (
            None
        )
        clamp_4 = torch.clamp(reshape_22, max=4.605170185988092)
        reshape_22 = None
        logit_scale_4 = clamp_4.exp()
        clamp_4 = None
        attn_27 = attn_26 * logit_scale_4
        attn_26 = logit_scale_4 = None
        x_111 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_112 = torch.nn.functional.relu(x_111, inplace=False)
        x_111 = None
        x_113 = torch.nn.functional.dropout(x_112, 0.125, False, False)
        x_112 = None
        x_114 = torch._C._nn.linear(
            x_113,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_113 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_115 = torch.nn.functional.dropout(x_114, 0.0, False, False)
        x_114 = None
        transpose_13 = x_115.transpose(1, 0)
        x_115 = None
        relative_position_bias_8 = transpose_13.reshape(12, 49, 49)
        transpose_13 = None
        relative_position_bias_9 = relative_position_bias_8.unsqueeze(0)
        relative_position_bias_8 = None
        attn_28 = attn_27 + relative_position_bias_9
        attn_27 = relative_position_bias_9 = None
        attn_29 = attn_28.softmax(dim=-1)
        attn_28 = None
        attn_30 = torch.nn.functional.dropout(attn_29, 0.0, False, False)
        attn_29 = None
        matmul_9 = attn_30 @ value_4
        attn_30 = value_4 = None
        transpose_14 = matmul_9.transpose(1, 2)
        matmul_9 = None
        x_116 = transpose_14.reshape(4, 49, -1)
        transpose_14 = None
        x_117 = torch._C._nn.linear(
            x_116,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_116 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_118 = torch.nn.functional.dropout(x_117, 0.0, False, False)
        x_117 = None
        attn_windows_4 = x_118.view(-1, 7, 7, 384)
        x_118 = None
        x_119 = attn_windows_4.view(-1, 2, 2, 7, 7, 384)
        attn_windows_4 = None
        permute_23 = x_119.permute(0, 1, 3, 2, 4, 5)
        x_119 = None
        contiguous_13 = permute_23.contiguous()
        permute_23 = None
        x_120 = contiguous_13.view(-1, 14, 14, 384)
        contiguous_13 = None
        getitem_23 = x_120[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_120 = None
        x_121 = getitem_23.contiguous()
        getitem_23 = None
        layer_norm_13 = torch.nn.functional.layer_norm(
            x_121,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_121 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (None)
        x_122 = x_108 + layer_norm_13
        x_108 = layer_norm_13 = None
        x_123 = x_122.reshape(1, -1, 384)
        x_122 = None
        x_124 = torch._C._nn.linear(
            x_123,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_125 = torch._C._nn.gelu(x_124, approximate="none")
        x_124 = None
        x_126 = torch.nn.functional.dropout(x_125, 0.0, False, False)
        x_125 = None
        x_127 = torch._C._nn.linear(
            x_126,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_126 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_128 = torch.nn.functional.dropout(x_127, 0.0, False, False)
        x_127 = None
        layer_norm_14 = torch.nn.functional.layer_norm(
            x_128,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_128 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_129 = x_123 + layer_norm_14
        x_123 = layer_norm_14 = None
        x_130 = x_129.reshape(1, 14, 14, 384)
        x_129 = None
        x_131 = torch.roll(x_130, shifts=(-3, -3), dims=(1, 2))
        x_132 = torch._C._nn.pad(x_131, (0, 0, 0, 0, 0, 0), "constant", None)
        x_131 = None
        x_133 = x_132.view(1, 2, 7, 2, 7, 384)
        x_132 = None
        permute_24 = x_133.permute(0, 1, 3, 2, 4, 5)
        x_133 = None
        contiguous_15 = permute_24.contiguous()
        permute_24 = None
        windows_5 = contiguous_15.view(-1, 7, 7, 384)
        contiguous_15 = None
        x_windows_5 = windows_5.view(-1, 49, 384)
        windows_5 = None
        linear_32 = torch._C._nn.linear(
            x_windows_5,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_,
        )
        x_windows_5 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_42 = linear_32.view(4, 49, 3, 12, 32)
        linear_32 = None
        qkv_5 = view_42.permute(2, 0, 3, 1, 4)
        view_42 = None
        unbind_5 = qkv_5.unbind(0)
        qkv_5 = None
        query_5 = unbind_5[0]
        key_5 = unbind_5[1]
        value_5 = unbind_5[2]
        unbind_5 = None
        normalize_10 = torch.nn.functional.normalize(query_5, dim=-1)
        query_5 = None
        normalize_11 = torch.nn.functional.normalize(key_5, dim=-1)
        key_5 = None
        transpose_15 = normalize_11.transpose(-2, -1)
        normalize_11 = None
        attn_31 = normalize_10 @ transpose_15
        normalize_10 = transpose_15 = None
        reshape_27 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_parameters_logit_scale_.reshape(
            1, 12, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_parameters_logit_scale_ = (
            None
        )
        clamp_5 = torch.clamp(reshape_27, max=4.605170185988092)
        reshape_27 = None
        logit_scale_5 = clamp_5.exp()
        clamp_5 = None
        attn_32 = attn_31 * logit_scale_5
        attn_31 = logit_scale_5 = None
        x_134 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_135 = torch.nn.functional.relu(x_134, inplace=False)
        x_134 = None
        x_136 = torch.nn.functional.dropout(x_135, 0.125, False, False)
        x_135 = None
        x_137 = torch._C._nn.linear(
            x_136,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_136 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_138 = torch.nn.functional.dropout(x_137, 0.0, False, False)
        x_137 = None
        transpose_16 = x_138.transpose(1, 0)
        x_138 = None
        relative_position_bias_10 = transpose_16.reshape(12, 49, 49)
        transpose_16 = None
        relative_position_bias_11 = relative_position_bias_10.unsqueeze(0)
        relative_position_bias_10 = None
        attn_33 = attn_32 + relative_position_bias_11
        attn_32 = relative_position_bias_11 = None
        attn_34 = attn_33.view(1, 4, 12, 49, 49)
        attn_33 = None
        unsqueeze_10 = l_self_modules_stages_modules_2_modules_blocks_modules_1_buffers_attn_mask_.unsqueeze(
            1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_buffers_attn_mask_ = (
            None
        )
        unsqueeze_11 = unsqueeze_10.unsqueeze(0)
        unsqueeze_10 = None
        attn_35 = attn_34 + unsqueeze_11
        attn_34 = unsqueeze_11 = None
        attn_36 = attn_35.view(-1, 12, 49, 49)
        attn_35 = None
        attn_37 = attn_36.softmax(dim=-1)
        attn_36 = None
        attn_38 = torch.nn.functional.dropout(attn_37, 0.0, False, False)
        attn_37 = None
        matmul_11 = attn_38 @ value_5
        attn_38 = value_5 = None
        transpose_17 = matmul_11.transpose(1, 2)
        matmul_11 = None
        x_139 = transpose_17.reshape(4, 49, -1)
        transpose_17 = None
        x_140 = torch._C._nn.linear(
            x_139,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_139 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_141 = torch.nn.functional.dropout(x_140, 0.0, False, False)
        x_140 = None
        attn_windows_5 = x_141.view(-1, 7, 7, 384)
        x_141 = None
        x_142 = attn_windows_5.view(-1, 2, 2, 7, 7, 384)
        attn_windows_5 = None
        permute_26 = x_142.permute(0, 1, 3, 2, 4, 5)
        x_142 = None
        contiguous_16 = permute_26.contiguous()
        permute_26 = None
        x_143 = contiguous_16.view(-1, 14, 14, 384)
        contiguous_16 = None
        getitem_27 = x_143[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_143 = None
        x_144 = getitem_27.contiguous()
        getitem_27 = None
        x_145 = torch.roll(x_144, shifts=(3, 3), dims=(1, 2))
        x_144 = None
        layer_norm_15 = torch.nn.functional.layer_norm(
            x_145,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_145 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_146 = x_130 + layer_norm_15
        x_130 = layer_norm_15 = None
        x_147 = x_146.reshape(1, -1, 384)
        x_146 = None
        x_148 = torch._C._nn.linear(
            x_147,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_149 = torch._C._nn.gelu(x_148, approximate="none")
        x_148 = None
        x_150 = torch.nn.functional.dropout(x_149, 0.0, False, False)
        x_149 = None
        x_151 = torch._C._nn.linear(
            x_150,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_150 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_152 = torch.nn.functional.dropout(x_151, 0.0, False, False)
        x_151 = None
        layer_norm_16 = torch.nn.functional.layer_norm(
            x_152,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_152 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_153 = x_147 + layer_norm_16
        x_147 = layer_norm_16 = None
        x_154 = x_153.reshape(1, 14, 14, 384)
        x_153 = None
        x_155 = torch._C._nn.pad(x_154, (0, 0, 0, 0, 0, 0), "constant", None)
        x_156 = x_155.view(1, 2, 7, 2, 7, 384)
        x_155 = None
        permute_27 = x_156.permute(0, 1, 3, 2, 4, 5)
        x_156 = None
        contiguous_18 = permute_27.contiguous()
        permute_27 = None
        windows_6 = contiguous_18.view(-1, 7, 7, 384)
        contiguous_18 = None
        x_windows_6 = windows_6.view(-1, 49, 384)
        windows_6 = None
        linear_38 = torch._C._nn.linear(
            x_windows_6,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_,
        )
        x_windows_6 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_51 = linear_38.view(4, 49, 3, 12, 32)
        linear_38 = None
        qkv_6 = view_51.permute(2, 0, 3, 1, 4)
        view_51 = None
        unbind_6 = qkv_6.unbind(0)
        qkv_6 = None
        query_6 = unbind_6[0]
        key_6 = unbind_6[1]
        value_6 = unbind_6[2]
        unbind_6 = None
        normalize_12 = torch.nn.functional.normalize(query_6, dim=-1)
        query_6 = None
        normalize_13 = torch.nn.functional.normalize(key_6, dim=-1)
        key_6 = None
        transpose_18 = normalize_13.transpose(-2, -1)
        normalize_13 = None
        attn_39 = normalize_12 @ transpose_18
        normalize_12 = transpose_18 = None
        reshape_32 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_parameters_logit_scale_.reshape(
            1, 12, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_parameters_logit_scale_ = (
            None
        )
        clamp_6 = torch.clamp(reshape_32, max=4.605170185988092)
        reshape_32 = None
        logit_scale_6 = clamp_6.exp()
        clamp_6 = None
        attn_40 = attn_39 * logit_scale_6
        attn_39 = logit_scale_6 = None
        x_157 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_158 = torch.nn.functional.relu(x_157, inplace=False)
        x_157 = None
        x_159 = torch.nn.functional.dropout(x_158, 0.125, False, False)
        x_158 = None
        x_160 = torch._C._nn.linear(
            x_159,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_159 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_161 = torch.nn.functional.dropout(x_160, 0.0, False, False)
        x_160 = None
        transpose_19 = x_161.transpose(1, 0)
        x_161 = None
        relative_position_bias_12 = transpose_19.reshape(12, 49, 49)
        transpose_19 = None
        relative_position_bias_13 = relative_position_bias_12.unsqueeze(0)
        relative_position_bias_12 = None
        attn_41 = attn_40 + relative_position_bias_13
        attn_40 = relative_position_bias_13 = None
        attn_42 = attn_41.softmax(dim=-1)
        attn_41 = None
        attn_43 = torch.nn.functional.dropout(attn_42, 0.0, False, False)
        attn_42 = None
        matmul_13 = attn_43 @ value_6
        attn_43 = value_6 = None
        transpose_20 = matmul_13.transpose(1, 2)
        matmul_13 = None
        x_162 = transpose_20.reshape(4, 49, -1)
        transpose_20 = None
        x_163 = torch._C._nn.linear(
            x_162,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_,
        )
        x_162 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_ = (None)
        x_164 = torch.nn.functional.dropout(x_163, 0.0, False, False)
        x_163 = None
        attn_windows_6 = x_164.view(-1, 7, 7, 384)
        x_164 = None
        x_165 = attn_windows_6.view(-1, 2, 2, 7, 7, 384)
        attn_windows_6 = None
        permute_29 = x_165.permute(0, 1, 3, 2, 4, 5)
        x_165 = None
        contiguous_19 = permute_29.contiguous()
        permute_29 = None
        x_166 = contiguous_19.view(-1, 14, 14, 384)
        contiguous_19 = None
        getitem_31 = x_166[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_166 = None
        x_167 = getitem_31.contiguous()
        getitem_31 = None
        layer_norm_17 = torch.nn.functional.layer_norm(
            x_167,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_167 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_ = (None)
        x_168 = x_154 + layer_norm_17
        x_154 = layer_norm_17 = None
        x_169 = x_168.reshape(1, -1, 384)
        x_168 = None
        x_170 = torch._C._nn.linear(
            x_169,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_171 = torch._C._nn.gelu(x_170, approximate="none")
        x_170 = None
        x_172 = torch.nn.functional.dropout(x_171, 0.0, False, False)
        x_171 = None
        x_173 = torch._C._nn.linear(
            x_172,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_172 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_174 = torch.nn.functional.dropout(x_173, 0.0, False, False)
        x_173 = None
        layer_norm_18 = torch.nn.functional.layer_norm(
            x_174,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_174 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_ = (None)
        x_175 = x_169 + layer_norm_18
        x_169 = layer_norm_18 = None
        x_176 = x_175.reshape(1, 14, 14, 384)
        x_175 = None
        x_177 = torch.roll(x_176, shifts=(-3, -3), dims=(1, 2))
        x_178 = torch._C._nn.pad(x_177, (0, 0, 0, 0, 0, 0), "constant", None)
        x_177 = None
        x_179 = x_178.view(1, 2, 7, 2, 7, 384)
        x_178 = None
        permute_30 = x_179.permute(0, 1, 3, 2, 4, 5)
        x_179 = None
        contiguous_21 = permute_30.contiguous()
        permute_30 = None
        windows_7 = contiguous_21.view(-1, 7, 7, 384)
        contiguous_21 = None
        x_windows_7 = windows_7.view(-1, 49, 384)
        windows_7 = None
        linear_44 = torch._C._nn.linear(
            x_windows_7,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_,
        )
        x_windows_7 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_58 = linear_44.view(4, 49, 3, 12, 32)
        linear_44 = None
        qkv_7 = view_58.permute(2, 0, 3, 1, 4)
        view_58 = None
        unbind_7 = qkv_7.unbind(0)
        qkv_7 = None
        query_7 = unbind_7[0]
        key_7 = unbind_7[1]
        value_7 = unbind_7[2]
        unbind_7 = None
        normalize_14 = torch.nn.functional.normalize(query_7, dim=-1)
        query_7 = None
        normalize_15 = torch.nn.functional.normalize(key_7, dim=-1)
        key_7 = None
        transpose_21 = normalize_15.transpose(-2, -1)
        normalize_15 = None
        attn_44 = normalize_14 @ transpose_21
        normalize_14 = transpose_21 = None
        reshape_37 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_parameters_logit_scale_.reshape(
            1, 12, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_parameters_logit_scale_ = (
            None
        )
        clamp_7 = torch.clamp(reshape_37, max=4.605170185988092)
        reshape_37 = None
        logit_scale_7 = clamp_7.exp()
        clamp_7 = None
        attn_45 = attn_44 * logit_scale_7
        attn_44 = logit_scale_7 = None
        x_180 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_181 = torch.nn.functional.relu(x_180, inplace=False)
        x_180 = None
        x_182 = torch.nn.functional.dropout(x_181, 0.125, False, False)
        x_181 = None
        x_183 = torch._C._nn.linear(
            x_182,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_182 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_184 = torch.nn.functional.dropout(x_183, 0.0, False, False)
        x_183 = None
        transpose_22 = x_184.transpose(1, 0)
        x_184 = None
        relative_position_bias_14 = transpose_22.reshape(12, 49, 49)
        transpose_22 = None
        relative_position_bias_15 = relative_position_bias_14.unsqueeze(0)
        relative_position_bias_14 = None
        attn_46 = attn_45 + relative_position_bias_15
        attn_45 = relative_position_bias_15 = None
        attn_47 = attn_46.view(1, 4, 12, 49, 49)
        attn_46 = None
        unsqueeze_14 = l_self_modules_stages_modules_2_modules_blocks_modules_3_buffers_attn_mask_.unsqueeze(
            1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_buffers_attn_mask_ = (
            None
        )
        unsqueeze_15 = unsqueeze_14.unsqueeze(0)
        unsqueeze_14 = None
        attn_48 = attn_47 + unsqueeze_15
        attn_47 = unsqueeze_15 = None
        attn_49 = attn_48.view(-1, 12, 49, 49)
        attn_48 = None
        attn_50 = attn_49.softmax(dim=-1)
        attn_49 = None
        attn_51 = torch.nn.functional.dropout(attn_50, 0.0, False, False)
        attn_50 = None
        matmul_15 = attn_51 @ value_7
        attn_51 = value_7 = None
        transpose_23 = matmul_15.transpose(1, 2)
        matmul_15 = None
        x_185 = transpose_23.reshape(4, 49, -1)
        transpose_23 = None
        x_186 = torch._C._nn.linear(
            x_185,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_,
        )
        x_185 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_ = (None)
        x_187 = torch.nn.functional.dropout(x_186, 0.0, False, False)
        x_186 = None
        attn_windows_7 = x_187.view(-1, 7, 7, 384)
        x_187 = None
        x_188 = attn_windows_7.view(-1, 2, 2, 7, 7, 384)
        attn_windows_7 = None
        permute_32 = x_188.permute(0, 1, 3, 2, 4, 5)
        x_188 = None
        contiguous_22 = permute_32.contiguous()
        permute_32 = None
        x_189 = contiguous_22.view(-1, 14, 14, 384)
        contiguous_22 = None
        getitem_35 = x_189[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_189 = None
        x_190 = getitem_35.contiguous()
        getitem_35 = None
        x_191 = torch.roll(x_190, shifts=(3, 3), dims=(1, 2))
        x_190 = None
        layer_norm_19 = torch.nn.functional.layer_norm(
            x_191,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_191 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_ = (None)
        x_192 = x_176 + layer_norm_19
        x_176 = layer_norm_19 = None
        x_193 = x_192.reshape(1, -1, 384)
        x_192 = None
        x_194 = torch._C._nn.linear(
            x_193,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_195 = torch._C._nn.gelu(x_194, approximate="none")
        x_194 = None
        x_196 = torch.nn.functional.dropout(x_195, 0.0, False, False)
        x_195 = None
        x_197 = torch._C._nn.linear(
            x_196,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_196 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_198 = torch.nn.functional.dropout(x_197, 0.0, False, False)
        x_197 = None
        layer_norm_20 = torch.nn.functional.layer_norm(
            x_198,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_198 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_ = (None)
        x_199 = x_193 + layer_norm_20
        x_193 = layer_norm_20 = None
        x_200 = x_199.reshape(1, 14, 14, 384)
        x_199 = None
        x_201 = torch._C._nn.pad(x_200, (0, 0, 0, 0, 0, 0), "constant", None)
        x_202 = x_201.view(1, 2, 7, 2, 7, 384)
        x_201 = None
        permute_33 = x_202.permute(0, 1, 3, 2, 4, 5)
        x_202 = None
        contiguous_24 = permute_33.contiguous()
        permute_33 = None
        windows_8 = contiguous_24.view(-1, 7, 7, 384)
        contiguous_24 = None
        x_windows_8 = windows_8.view(-1, 49, 384)
        windows_8 = None
        linear_50 = torch._C._nn.linear(
            x_windows_8,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_,
        )
        x_windows_8 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_67 = linear_50.view(4, 49, 3, 12, 32)
        linear_50 = None
        qkv_8 = view_67.permute(2, 0, 3, 1, 4)
        view_67 = None
        unbind_8 = qkv_8.unbind(0)
        qkv_8 = None
        query_8 = unbind_8[0]
        key_8 = unbind_8[1]
        value_8 = unbind_8[2]
        unbind_8 = None
        normalize_16 = torch.nn.functional.normalize(query_8, dim=-1)
        query_8 = None
        normalize_17 = torch.nn.functional.normalize(key_8, dim=-1)
        key_8 = None
        transpose_24 = normalize_17.transpose(-2, -1)
        normalize_17 = None
        attn_52 = normalize_16 @ transpose_24
        normalize_16 = transpose_24 = None
        reshape_42 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_parameters_logit_scale_.reshape(
            1, 12, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_parameters_logit_scale_ = (
            None
        )
        clamp_8 = torch.clamp(reshape_42, max=4.605170185988092)
        reshape_42 = None
        logit_scale_8 = clamp_8.exp()
        clamp_8 = None
        attn_53 = attn_52 * logit_scale_8
        attn_52 = logit_scale_8 = None
        x_203 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_204 = torch.nn.functional.relu(x_203, inplace=False)
        x_203 = None
        x_205 = torch.nn.functional.dropout(x_204, 0.125, False, False)
        x_204 = None
        x_206 = torch._C._nn.linear(
            x_205,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_205 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_207 = torch.nn.functional.dropout(x_206, 0.0, False, False)
        x_206 = None
        transpose_25 = x_207.transpose(1, 0)
        x_207 = None
        relative_position_bias_16 = transpose_25.reshape(12, 49, 49)
        transpose_25 = None
        relative_position_bias_17 = relative_position_bias_16.unsqueeze(0)
        relative_position_bias_16 = None
        attn_54 = attn_53 + relative_position_bias_17
        attn_53 = relative_position_bias_17 = None
        attn_55 = attn_54.softmax(dim=-1)
        attn_54 = None
        attn_56 = torch.nn.functional.dropout(attn_55, 0.0, False, False)
        attn_55 = None
        matmul_17 = attn_56 @ value_8
        attn_56 = value_8 = None
        transpose_26 = matmul_17.transpose(1, 2)
        matmul_17 = None
        x_208 = transpose_26.reshape(4, 49, -1)
        transpose_26 = None
        x_209 = torch._C._nn.linear(
            x_208,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_,
        )
        x_208 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_ = (None)
        x_210 = torch.nn.functional.dropout(x_209, 0.0, False, False)
        x_209 = None
        attn_windows_8 = x_210.view(-1, 7, 7, 384)
        x_210 = None
        x_211 = attn_windows_8.view(-1, 2, 2, 7, 7, 384)
        attn_windows_8 = None
        permute_35 = x_211.permute(0, 1, 3, 2, 4, 5)
        x_211 = None
        contiguous_25 = permute_35.contiguous()
        permute_35 = None
        x_212 = contiguous_25.view(-1, 14, 14, 384)
        contiguous_25 = None
        getitem_39 = x_212[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_212 = None
        x_213 = getitem_39.contiguous()
        getitem_39 = None
        layer_norm_21 = torch.nn.functional.layer_norm(
            x_213,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_213 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_ = (None)
        x_214 = x_200 + layer_norm_21
        x_200 = layer_norm_21 = None
        x_215 = x_214.reshape(1, -1, 384)
        x_214 = None
        x_216 = torch._C._nn.linear(
            x_215,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_217 = torch._C._nn.gelu(x_216, approximate="none")
        x_216 = None
        x_218 = torch.nn.functional.dropout(x_217, 0.0, False, False)
        x_217 = None
        x_219 = torch._C._nn.linear(
            x_218,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_218 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_220 = torch.nn.functional.dropout(x_219, 0.0, False, False)
        x_219 = None
        layer_norm_22 = torch.nn.functional.layer_norm(
            x_220,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_220 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_ = (None)
        x_221 = x_215 + layer_norm_22
        x_215 = layer_norm_22 = None
        x_222 = x_221.reshape(1, 14, 14, 384)
        x_221 = None
        x_223 = torch.roll(x_222, shifts=(-3, -3), dims=(1, 2))
        x_224 = torch._C._nn.pad(x_223, (0, 0, 0, 0, 0, 0), "constant", None)
        x_223 = None
        x_225 = x_224.view(1, 2, 7, 2, 7, 384)
        x_224 = None
        permute_36 = x_225.permute(0, 1, 3, 2, 4, 5)
        x_225 = None
        contiguous_27 = permute_36.contiguous()
        permute_36 = None
        windows_9 = contiguous_27.view(-1, 7, 7, 384)
        contiguous_27 = None
        x_windows_9 = windows_9.view(-1, 49, 384)
        windows_9 = None
        linear_56 = torch._C._nn.linear(
            x_windows_9,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_,
        )
        x_windows_9 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_74 = linear_56.view(4, 49, 3, 12, 32)
        linear_56 = None
        qkv_9 = view_74.permute(2, 0, 3, 1, 4)
        view_74 = None
        unbind_9 = qkv_9.unbind(0)
        qkv_9 = None
        query_9 = unbind_9[0]
        key_9 = unbind_9[1]
        value_9 = unbind_9[2]
        unbind_9 = None
        normalize_18 = torch.nn.functional.normalize(query_9, dim=-1)
        query_9 = None
        normalize_19 = torch.nn.functional.normalize(key_9, dim=-1)
        key_9 = None
        transpose_27 = normalize_19.transpose(-2, -1)
        normalize_19 = None
        attn_57 = normalize_18 @ transpose_27
        normalize_18 = transpose_27 = None
        reshape_47 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_parameters_logit_scale_.reshape(
            1, 12, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_parameters_logit_scale_ = (
            None
        )
        clamp_9 = torch.clamp(reshape_47, max=4.605170185988092)
        reshape_47 = None
        logit_scale_9 = clamp_9.exp()
        clamp_9 = None
        attn_58 = attn_57 * logit_scale_9
        attn_57 = logit_scale_9 = None
        x_226 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_227 = torch.nn.functional.relu(x_226, inplace=False)
        x_226 = None
        x_228 = torch.nn.functional.dropout(x_227, 0.125, False, False)
        x_227 = None
        x_229 = torch._C._nn.linear(
            x_228,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_228 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_230 = torch.nn.functional.dropout(x_229, 0.0, False, False)
        x_229 = None
        transpose_28 = x_230.transpose(1, 0)
        x_230 = None
        relative_position_bias_18 = transpose_28.reshape(12, 49, 49)
        transpose_28 = None
        relative_position_bias_19 = relative_position_bias_18.unsqueeze(0)
        relative_position_bias_18 = None
        attn_59 = attn_58 + relative_position_bias_19
        attn_58 = relative_position_bias_19 = None
        attn_60 = attn_59.view(1, 4, 12, 49, 49)
        attn_59 = None
        unsqueeze_18 = l_self_modules_stages_modules_2_modules_blocks_modules_5_buffers_attn_mask_.unsqueeze(
            1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_buffers_attn_mask_ = (
            None
        )
        unsqueeze_19 = unsqueeze_18.unsqueeze(0)
        unsqueeze_18 = None
        attn_61 = attn_60 + unsqueeze_19
        attn_60 = unsqueeze_19 = None
        attn_62 = attn_61.view(-1, 12, 49, 49)
        attn_61 = None
        attn_63 = attn_62.softmax(dim=-1)
        attn_62 = None
        attn_64 = torch.nn.functional.dropout(attn_63, 0.0, False, False)
        attn_63 = None
        matmul_19 = attn_64 @ value_9
        attn_64 = value_9 = None
        transpose_29 = matmul_19.transpose(1, 2)
        matmul_19 = None
        x_231 = transpose_29.reshape(4, 49, -1)
        transpose_29 = None
        x_232 = torch._C._nn.linear(
            x_231,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_,
        )
        x_231 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_ = (None)
        x_233 = torch.nn.functional.dropout(x_232, 0.0, False, False)
        x_232 = None
        attn_windows_9 = x_233.view(-1, 7, 7, 384)
        x_233 = None
        x_234 = attn_windows_9.view(-1, 2, 2, 7, 7, 384)
        attn_windows_9 = None
        permute_38 = x_234.permute(0, 1, 3, 2, 4, 5)
        x_234 = None
        contiguous_28 = permute_38.contiguous()
        permute_38 = None
        x_235 = contiguous_28.view(-1, 14, 14, 384)
        contiguous_28 = None
        getitem_43 = x_235[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_235 = None
        x_236 = getitem_43.contiguous()
        getitem_43 = None
        x_237 = torch.roll(x_236, shifts=(3, 3), dims=(1, 2))
        x_236 = None
        layer_norm_23 = torch.nn.functional.layer_norm(
            x_237,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_237 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_ = (None)
        x_238 = x_222 + layer_norm_23
        x_222 = layer_norm_23 = None
        x_239 = x_238.reshape(1, -1, 384)
        x_238 = None
        x_240 = torch._C._nn.linear(
            x_239,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_241 = torch._C._nn.gelu(x_240, approximate="none")
        x_240 = None
        x_242 = torch.nn.functional.dropout(x_241, 0.0, False, False)
        x_241 = None
        x_243 = torch._C._nn.linear(
            x_242,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_242 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_244 = torch.nn.functional.dropout(x_243, 0.0, False, False)
        x_243 = None
        layer_norm_24 = torch.nn.functional.layer_norm(
            x_244,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_244 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_ = (None)
        x_245 = x_239 + layer_norm_24
        x_239 = layer_norm_24 = None
        x_246 = x_245.reshape(1, 14, 14, 384)
        x_245 = None
        x_247 = torch._C._nn.pad(x_246, (0, 0, 0, 0, 0, 0), "constant", None)
        x_248 = x_247.view(1, 2, 7, 2, 7, 384)
        x_247 = None
        permute_39 = x_248.permute(0, 1, 3, 2, 4, 5)
        x_248 = None
        contiguous_30 = permute_39.contiguous()
        permute_39 = None
        windows_10 = contiguous_30.view(-1, 7, 7, 384)
        contiguous_30 = None
        x_windows_10 = windows_10.view(-1, 49, 384)
        windows_10 = None
        linear_62 = torch._C._nn.linear(
            x_windows_10,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_,
        )
        x_windows_10 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_83 = linear_62.view(4, 49, 3, 12, 32)
        linear_62 = None
        qkv_10 = view_83.permute(2, 0, 3, 1, 4)
        view_83 = None
        unbind_10 = qkv_10.unbind(0)
        qkv_10 = None
        query_10 = unbind_10[0]
        key_10 = unbind_10[1]
        value_10 = unbind_10[2]
        unbind_10 = None
        normalize_20 = torch.nn.functional.normalize(query_10, dim=-1)
        query_10 = None
        normalize_21 = torch.nn.functional.normalize(key_10, dim=-1)
        key_10 = None
        transpose_30 = normalize_21.transpose(-2, -1)
        normalize_21 = None
        attn_65 = normalize_20 @ transpose_30
        normalize_20 = transpose_30 = None
        reshape_52 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_parameters_logit_scale_.reshape(
            1, 12, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_parameters_logit_scale_ = (
            None
        )
        clamp_10 = torch.clamp(reshape_52, max=4.605170185988092)
        reshape_52 = None
        logit_scale_10 = clamp_10.exp()
        clamp_10 = None
        attn_66 = attn_65 * logit_scale_10
        attn_65 = logit_scale_10 = None
        x_249 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_250 = torch.nn.functional.relu(x_249, inplace=False)
        x_249 = None
        x_251 = torch.nn.functional.dropout(x_250, 0.125, False, False)
        x_250 = None
        x_252 = torch._C._nn.linear(
            x_251,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_251 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_253 = torch.nn.functional.dropout(x_252, 0.0, False, False)
        x_252 = None
        transpose_31 = x_253.transpose(1, 0)
        x_253 = None
        relative_position_bias_20 = transpose_31.reshape(12, 49, 49)
        transpose_31 = None
        relative_position_bias_21 = relative_position_bias_20.unsqueeze(0)
        relative_position_bias_20 = None
        attn_67 = attn_66 + relative_position_bias_21
        attn_66 = relative_position_bias_21 = None
        attn_68 = attn_67.softmax(dim=-1)
        attn_67 = None
        attn_69 = torch.nn.functional.dropout(attn_68, 0.0, False, False)
        attn_68 = None
        matmul_21 = attn_69 @ value_10
        attn_69 = value_10 = None
        transpose_32 = matmul_21.transpose(1, 2)
        matmul_21 = None
        x_254 = transpose_32.reshape(4, 49, -1)
        transpose_32 = None
        x_255 = torch._C._nn.linear(
            x_254,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_,
        )
        x_254 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_ = (None)
        x_256 = torch.nn.functional.dropout(x_255, 0.0, False, False)
        x_255 = None
        attn_windows_10 = x_256.view(-1, 7, 7, 384)
        x_256 = None
        x_257 = attn_windows_10.view(-1, 2, 2, 7, 7, 384)
        attn_windows_10 = None
        permute_41 = x_257.permute(0, 1, 3, 2, 4, 5)
        x_257 = None
        contiguous_31 = permute_41.contiguous()
        permute_41 = None
        x_258 = contiguous_31.view(-1, 14, 14, 384)
        contiguous_31 = None
        getitem_47 = x_258[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_258 = None
        x_259 = getitem_47.contiguous()
        getitem_47 = None
        layer_norm_25 = torch.nn.functional.layer_norm(
            x_259,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_259 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_bias_ = (None)
        x_260 = x_246 + layer_norm_25
        x_246 = layer_norm_25 = None
        x_261 = x_260.reshape(1, -1, 384)
        x_260 = None
        x_262 = torch._C._nn.linear(
            x_261,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_263 = torch._C._nn.gelu(x_262, approximate="none")
        x_262 = None
        x_264 = torch.nn.functional.dropout(x_263, 0.0, False, False)
        x_263 = None
        x_265 = torch._C._nn.linear(
            x_264,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_264 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_266 = torch.nn.functional.dropout(x_265, 0.0, False, False)
        x_265 = None
        layer_norm_26 = torch.nn.functional.layer_norm(
            x_266,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_266 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_bias_ = (None)
        x_267 = x_261 + layer_norm_26
        x_261 = layer_norm_26 = None
        x_268 = x_267.reshape(1, 14, 14, 384)
        x_267 = None
        x_269 = torch.roll(x_268, shifts=(-3, -3), dims=(1, 2))
        x_270 = torch._C._nn.pad(x_269, (0, 0, 0, 0, 0, 0), "constant", None)
        x_269 = None
        x_271 = x_270.view(1, 2, 7, 2, 7, 384)
        x_270 = None
        permute_42 = x_271.permute(0, 1, 3, 2, 4, 5)
        x_271 = None
        contiguous_33 = permute_42.contiguous()
        permute_42 = None
        windows_11 = contiguous_33.view(-1, 7, 7, 384)
        contiguous_33 = None
        x_windows_11 = windows_11.view(-1, 49, 384)
        windows_11 = None
        linear_68 = torch._C._nn.linear(
            x_windows_11,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_,
        )
        x_windows_11 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_90 = linear_68.view(4, 49, 3, 12, 32)
        linear_68 = None
        qkv_11 = view_90.permute(2, 0, 3, 1, 4)
        view_90 = None
        unbind_11 = qkv_11.unbind(0)
        qkv_11 = None
        query_11 = unbind_11[0]
        key_11 = unbind_11[1]
        value_11 = unbind_11[2]
        unbind_11 = None
        normalize_22 = torch.nn.functional.normalize(query_11, dim=-1)
        query_11 = None
        normalize_23 = torch.nn.functional.normalize(key_11, dim=-1)
        key_11 = None
        transpose_33 = normalize_23.transpose(-2, -1)
        normalize_23 = None
        attn_70 = normalize_22 @ transpose_33
        normalize_22 = transpose_33 = None
        reshape_57 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_parameters_logit_scale_.reshape(
            1, 12, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_parameters_logit_scale_ = (
            None
        )
        clamp_11 = torch.clamp(reshape_57, max=4.605170185988092)
        reshape_57 = None
        logit_scale_11 = clamp_11.exp()
        clamp_11 = None
        attn_71 = attn_70 * logit_scale_11
        attn_70 = logit_scale_11 = None
        x_272 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_273 = torch.nn.functional.relu(x_272, inplace=False)
        x_272 = None
        x_274 = torch.nn.functional.dropout(x_273, 0.125, False, False)
        x_273 = None
        x_275 = torch._C._nn.linear(
            x_274,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_274 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_276 = torch.nn.functional.dropout(x_275, 0.0, False, False)
        x_275 = None
        transpose_34 = x_276.transpose(1, 0)
        x_276 = None
        relative_position_bias_22 = transpose_34.reshape(12, 49, 49)
        transpose_34 = None
        relative_position_bias_23 = relative_position_bias_22.unsqueeze(0)
        relative_position_bias_22 = None
        attn_72 = attn_71 + relative_position_bias_23
        attn_71 = relative_position_bias_23 = None
        attn_73 = attn_72.view(1, 4, 12, 49, 49)
        attn_72 = None
        unsqueeze_22 = l_self_modules_stages_modules_2_modules_blocks_modules_7_buffers_attn_mask_.unsqueeze(
            1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_buffers_attn_mask_ = (
            None
        )
        unsqueeze_23 = unsqueeze_22.unsqueeze(0)
        unsqueeze_22 = None
        attn_74 = attn_73 + unsqueeze_23
        attn_73 = unsqueeze_23 = None
        attn_75 = attn_74.view(-1, 12, 49, 49)
        attn_74 = None
        attn_76 = attn_75.softmax(dim=-1)
        attn_75 = None
        attn_77 = torch.nn.functional.dropout(attn_76, 0.0, False, False)
        attn_76 = None
        matmul_23 = attn_77 @ value_11
        attn_77 = value_11 = None
        transpose_35 = matmul_23.transpose(1, 2)
        matmul_23 = None
        x_277 = transpose_35.reshape(4, 49, -1)
        transpose_35 = None
        x_278 = torch._C._nn.linear(
            x_277,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_,
        )
        x_277 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_ = (None)
        x_279 = torch.nn.functional.dropout(x_278, 0.0, False, False)
        x_278 = None
        attn_windows_11 = x_279.view(-1, 7, 7, 384)
        x_279 = None
        x_280 = attn_windows_11.view(-1, 2, 2, 7, 7, 384)
        attn_windows_11 = None
        permute_44 = x_280.permute(0, 1, 3, 2, 4, 5)
        x_280 = None
        contiguous_34 = permute_44.contiguous()
        permute_44 = None
        x_281 = contiguous_34.view(-1, 14, 14, 384)
        contiguous_34 = None
        getitem_51 = x_281[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_281 = None
        x_282 = getitem_51.contiguous()
        getitem_51 = None
        x_283 = torch.roll(x_282, shifts=(3, 3), dims=(1, 2))
        x_282 = None
        layer_norm_27 = torch.nn.functional.layer_norm(
            x_283,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_283 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_bias_ = (None)
        x_284 = x_268 + layer_norm_27
        x_268 = layer_norm_27 = None
        x_285 = x_284.reshape(1, -1, 384)
        x_284 = None
        x_286 = torch._C._nn.linear(
            x_285,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_287 = torch._C._nn.gelu(x_286, approximate="none")
        x_286 = None
        x_288 = torch.nn.functional.dropout(x_287, 0.0, False, False)
        x_287 = None
        x_289 = torch._C._nn.linear(
            x_288,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_288 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_290 = torch.nn.functional.dropout(x_289, 0.0, False, False)
        x_289 = None
        layer_norm_28 = torch.nn.functional.layer_norm(
            x_290,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_290 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_bias_ = (None)
        x_291 = x_285 + layer_norm_28
        x_285 = layer_norm_28 = None
        x_292 = x_291.reshape(1, 14, 14, 384)
        x_291 = None
        x_293 = torch._C._nn.pad(x_292, (0, 0, 0, 0, 0, 0), "constant", None)
        x_294 = x_293.view(1, 2, 7, 2, 7, 384)
        x_293 = None
        permute_45 = x_294.permute(0, 1, 3, 2, 4, 5)
        x_294 = None
        contiguous_36 = permute_45.contiguous()
        permute_45 = None
        windows_12 = contiguous_36.view(-1, 7, 7, 384)
        contiguous_36 = None
        x_windows_12 = windows_12.view(-1, 49, 384)
        windows_12 = None
        linear_74 = torch._C._nn.linear(
            x_windows_12,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_bias_,
        )
        x_windows_12 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_99 = linear_74.view(4, 49, 3, 12, 32)
        linear_74 = None
        qkv_12 = view_99.permute(2, 0, 3, 1, 4)
        view_99 = None
        unbind_12 = qkv_12.unbind(0)
        qkv_12 = None
        query_12 = unbind_12[0]
        key_12 = unbind_12[1]
        value_12 = unbind_12[2]
        unbind_12 = None
        normalize_24 = torch.nn.functional.normalize(query_12, dim=-1)
        query_12 = None
        normalize_25 = torch.nn.functional.normalize(key_12, dim=-1)
        key_12 = None
        transpose_36 = normalize_25.transpose(-2, -1)
        normalize_25 = None
        attn_78 = normalize_24 @ transpose_36
        normalize_24 = transpose_36 = None
        reshape_62 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_parameters_logit_scale_.reshape(
            1, 12, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_parameters_logit_scale_ = (
            None
        )
        clamp_12 = torch.clamp(reshape_62, max=4.605170185988092)
        reshape_62 = None
        logit_scale_12 = clamp_12.exp()
        clamp_12 = None
        attn_79 = attn_78 * logit_scale_12
        attn_78 = logit_scale_12 = None
        x_295 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_296 = torch.nn.functional.relu(x_295, inplace=False)
        x_295 = None
        x_297 = torch.nn.functional.dropout(x_296, 0.125, False, False)
        x_296 = None
        x_298 = torch._C._nn.linear(
            x_297,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_297 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_299 = torch.nn.functional.dropout(x_298, 0.0, False, False)
        x_298 = None
        transpose_37 = x_299.transpose(1, 0)
        x_299 = None
        relative_position_bias_24 = transpose_37.reshape(12, 49, 49)
        transpose_37 = None
        relative_position_bias_25 = relative_position_bias_24.unsqueeze(0)
        relative_position_bias_24 = None
        attn_80 = attn_79 + relative_position_bias_25
        attn_79 = relative_position_bias_25 = None
        attn_81 = attn_80.softmax(dim=-1)
        attn_80 = None
        attn_82 = torch.nn.functional.dropout(attn_81, 0.0, False, False)
        attn_81 = None
        matmul_25 = attn_82 @ value_12
        attn_82 = value_12 = None
        transpose_38 = matmul_25.transpose(1, 2)
        matmul_25 = None
        x_300 = transpose_38.reshape(4, 49, -1)
        transpose_38 = None
        x_301 = torch._C._nn.linear(
            x_300,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_,
        )
        x_300 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_ = (None)
        x_302 = torch.nn.functional.dropout(x_301, 0.0, False, False)
        x_301 = None
        attn_windows_12 = x_302.view(-1, 7, 7, 384)
        x_302 = None
        x_303 = attn_windows_12.view(-1, 2, 2, 7, 7, 384)
        attn_windows_12 = None
        permute_47 = x_303.permute(0, 1, 3, 2, 4, 5)
        x_303 = None
        contiguous_37 = permute_47.contiguous()
        permute_47 = None
        x_304 = contiguous_37.view(-1, 14, 14, 384)
        contiguous_37 = None
        getitem_55 = x_304[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_304 = None
        x_305 = getitem_55.contiguous()
        getitem_55 = None
        layer_norm_29 = torch.nn.functional.layer_norm(
            x_305,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_305 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_bias_ = (None)
        x_306 = x_292 + layer_norm_29
        x_292 = layer_norm_29 = None
        x_307 = x_306.reshape(1, -1, 384)
        x_306 = None
        x_308 = torch._C._nn.linear(
            x_307,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_309 = torch._C._nn.gelu(x_308, approximate="none")
        x_308 = None
        x_310 = torch.nn.functional.dropout(x_309, 0.0, False, False)
        x_309 = None
        x_311 = torch._C._nn.linear(
            x_310,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_310 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_312 = torch.nn.functional.dropout(x_311, 0.0, False, False)
        x_311 = None
        layer_norm_30 = torch.nn.functional.layer_norm(
            x_312,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_312 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_bias_ = (None)
        x_313 = x_307 + layer_norm_30
        x_307 = layer_norm_30 = None
        x_314 = x_313.reshape(1, 14, 14, 384)
        x_313 = None
        x_315 = torch.roll(x_314, shifts=(-3, -3), dims=(1, 2))
        x_316 = torch._C._nn.pad(x_315, (0, 0, 0, 0, 0, 0), "constant", None)
        x_315 = None
        x_317 = x_316.view(1, 2, 7, 2, 7, 384)
        x_316 = None
        permute_48 = x_317.permute(0, 1, 3, 2, 4, 5)
        x_317 = None
        contiguous_39 = permute_48.contiguous()
        permute_48 = None
        windows_13 = contiguous_39.view(-1, 7, 7, 384)
        contiguous_39 = None
        x_windows_13 = windows_13.view(-1, 49, 384)
        windows_13 = None
        linear_80 = torch._C._nn.linear(
            x_windows_13,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_bias_,
        )
        x_windows_13 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_106 = linear_80.view(4, 49, 3, 12, 32)
        linear_80 = None
        qkv_13 = view_106.permute(2, 0, 3, 1, 4)
        view_106 = None
        unbind_13 = qkv_13.unbind(0)
        qkv_13 = None
        query_13 = unbind_13[0]
        key_13 = unbind_13[1]
        value_13 = unbind_13[2]
        unbind_13 = None
        normalize_26 = torch.nn.functional.normalize(query_13, dim=-1)
        query_13 = None
        normalize_27 = torch.nn.functional.normalize(key_13, dim=-1)
        key_13 = None
        transpose_39 = normalize_27.transpose(-2, -1)
        normalize_27 = None
        attn_83 = normalize_26 @ transpose_39
        normalize_26 = transpose_39 = None
        reshape_67 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_parameters_logit_scale_.reshape(
            1, 12, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_parameters_logit_scale_ = (
            None
        )
        clamp_13 = torch.clamp(reshape_67, max=4.605170185988092)
        reshape_67 = None
        logit_scale_13 = clamp_13.exp()
        clamp_13 = None
        attn_84 = attn_83 * logit_scale_13
        attn_83 = logit_scale_13 = None
        x_318 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_319 = torch.nn.functional.relu(x_318, inplace=False)
        x_318 = None
        x_320 = torch.nn.functional.dropout(x_319, 0.125, False, False)
        x_319 = None
        x_321 = torch._C._nn.linear(
            x_320,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_320 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_322 = torch.nn.functional.dropout(x_321, 0.0, False, False)
        x_321 = None
        transpose_40 = x_322.transpose(1, 0)
        x_322 = None
        relative_position_bias_26 = transpose_40.reshape(12, 49, 49)
        transpose_40 = None
        relative_position_bias_27 = relative_position_bias_26.unsqueeze(0)
        relative_position_bias_26 = None
        attn_85 = attn_84 + relative_position_bias_27
        attn_84 = relative_position_bias_27 = None
        attn_86 = attn_85.view(1, 4, 12, 49, 49)
        attn_85 = None
        unsqueeze_26 = l_self_modules_stages_modules_2_modules_blocks_modules_9_buffers_attn_mask_.unsqueeze(
            1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_9_buffers_attn_mask_ = (
            None
        )
        unsqueeze_27 = unsqueeze_26.unsqueeze(0)
        unsqueeze_26 = None
        attn_87 = attn_86 + unsqueeze_27
        attn_86 = unsqueeze_27 = None
        attn_88 = attn_87.view(-1, 12, 49, 49)
        attn_87 = None
        attn_89 = attn_88.softmax(dim=-1)
        attn_88 = None
        attn_90 = torch.nn.functional.dropout(attn_89, 0.0, False, False)
        attn_89 = None
        matmul_27 = attn_90 @ value_13
        attn_90 = value_13 = None
        transpose_41 = matmul_27.transpose(1, 2)
        matmul_27 = None
        x_323 = transpose_41.reshape(4, 49, -1)
        transpose_41 = None
        x_324 = torch._C._nn.linear(
            x_323,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_,
        )
        x_323 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_ = (None)
        x_325 = torch.nn.functional.dropout(x_324, 0.0, False, False)
        x_324 = None
        attn_windows_13 = x_325.view(-1, 7, 7, 384)
        x_325 = None
        x_326 = attn_windows_13.view(-1, 2, 2, 7, 7, 384)
        attn_windows_13 = None
        permute_50 = x_326.permute(0, 1, 3, 2, 4, 5)
        x_326 = None
        contiguous_40 = permute_50.contiguous()
        permute_50 = None
        x_327 = contiguous_40.view(-1, 14, 14, 384)
        contiguous_40 = None
        getitem_59 = x_327[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_327 = None
        x_328 = getitem_59.contiguous()
        getitem_59 = None
        x_329 = torch.roll(x_328, shifts=(3, 3), dims=(1, 2))
        x_328 = None
        layer_norm_31 = torch.nn.functional.layer_norm(
            x_329,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_329 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_bias_ = (None)
        x_330 = x_314 + layer_norm_31
        x_314 = layer_norm_31 = None
        x_331 = x_330.reshape(1, -1, 384)
        x_330 = None
        x_332 = torch._C._nn.linear(
            x_331,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_333 = torch._C._nn.gelu(x_332, approximate="none")
        x_332 = None
        x_334 = torch.nn.functional.dropout(x_333, 0.0, False, False)
        x_333 = None
        x_335 = torch._C._nn.linear(
            x_334,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_334 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_336 = torch.nn.functional.dropout(x_335, 0.0, False, False)
        x_335 = None
        layer_norm_32 = torch.nn.functional.layer_norm(
            x_336,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_336 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_bias_ = (None)
        x_337 = x_331 + layer_norm_32
        x_331 = layer_norm_32 = None
        x_338 = x_337.reshape(1, 14, 14, 384)
        x_337 = None
        x_339 = torch._C._nn.pad(x_338, (0, 0, 0, 0, 0, 0), "constant", None)
        x_340 = x_339.view(1, 2, 7, 2, 7, 384)
        x_339 = None
        permute_51 = x_340.permute(0, 1, 3, 2, 4, 5)
        x_340 = None
        contiguous_42 = permute_51.contiguous()
        permute_51 = None
        windows_14 = contiguous_42.view(-1, 7, 7, 384)
        contiguous_42 = None
        x_windows_14 = windows_14.view(-1, 49, 384)
        windows_14 = None
        linear_86 = torch._C._nn.linear(
            x_windows_14,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_bias_,
        )
        x_windows_14 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_115 = linear_86.view(4, 49, 3, 12, 32)
        linear_86 = None
        qkv_14 = view_115.permute(2, 0, 3, 1, 4)
        view_115 = None
        unbind_14 = qkv_14.unbind(0)
        qkv_14 = None
        query_14 = unbind_14[0]
        key_14 = unbind_14[1]
        value_14 = unbind_14[2]
        unbind_14 = None
        normalize_28 = torch.nn.functional.normalize(query_14, dim=-1)
        query_14 = None
        normalize_29 = torch.nn.functional.normalize(key_14, dim=-1)
        key_14 = None
        transpose_42 = normalize_29.transpose(-2, -1)
        normalize_29 = None
        attn_91 = normalize_28 @ transpose_42
        normalize_28 = transpose_42 = None
        reshape_72 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_parameters_logit_scale_.reshape(
            1, 12, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_parameters_logit_scale_ = (
            None
        )
        clamp_14 = torch.clamp(reshape_72, max=4.605170185988092)
        reshape_72 = None
        logit_scale_14 = clamp_14.exp()
        clamp_14 = None
        attn_92 = attn_91 * logit_scale_14
        attn_91 = logit_scale_14 = None
        x_341 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_342 = torch.nn.functional.relu(x_341, inplace=False)
        x_341 = None
        x_343 = torch.nn.functional.dropout(x_342, 0.125, False, False)
        x_342 = None
        x_344 = torch._C._nn.linear(
            x_343,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_343 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_345 = torch.nn.functional.dropout(x_344, 0.0, False, False)
        x_344 = None
        transpose_43 = x_345.transpose(1, 0)
        x_345 = None
        relative_position_bias_28 = transpose_43.reshape(12, 49, 49)
        transpose_43 = None
        relative_position_bias_29 = relative_position_bias_28.unsqueeze(0)
        relative_position_bias_28 = None
        attn_93 = attn_92 + relative_position_bias_29
        attn_92 = relative_position_bias_29 = None
        attn_94 = attn_93.softmax(dim=-1)
        attn_93 = None
        attn_95 = torch.nn.functional.dropout(attn_94, 0.0, False, False)
        attn_94 = None
        matmul_29 = attn_95 @ value_14
        attn_95 = value_14 = None
        transpose_44 = matmul_29.transpose(1, 2)
        matmul_29 = None
        x_346 = transpose_44.reshape(4, 49, -1)
        transpose_44 = None
        x_347 = torch._C._nn.linear(
            x_346,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_,
        )
        x_346 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_ = (None)
        x_348 = torch.nn.functional.dropout(x_347, 0.0, False, False)
        x_347 = None
        attn_windows_14 = x_348.view(-1, 7, 7, 384)
        x_348 = None
        x_349 = attn_windows_14.view(-1, 2, 2, 7, 7, 384)
        attn_windows_14 = None
        permute_53 = x_349.permute(0, 1, 3, 2, 4, 5)
        x_349 = None
        contiguous_43 = permute_53.contiguous()
        permute_53 = None
        x_350 = contiguous_43.view(-1, 14, 14, 384)
        contiguous_43 = None
        getitem_63 = x_350[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_350 = None
        x_351 = getitem_63.contiguous()
        getitem_63 = None
        layer_norm_33 = torch.nn.functional.layer_norm(
            x_351,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_351 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_bias_ = (None)
        x_352 = x_338 + layer_norm_33
        x_338 = layer_norm_33 = None
        x_353 = x_352.reshape(1, -1, 384)
        x_352 = None
        x_354 = torch._C._nn.linear(
            x_353,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_355 = torch._C._nn.gelu(x_354, approximate="none")
        x_354 = None
        x_356 = torch.nn.functional.dropout(x_355, 0.0, False, False)
        x_355 = None
        x_357 = torch._C._nn.linear(
            x_356,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_356 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_358 = torch.nn.functional.dropout(x_357, 0.0, False, False)
        x_357 = None
        layer_norm_34 = torch.nn.functional.layer_norm(
            x_358,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_358 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_bias_ = (None)
        x_359 = x_353 + layer_norm_34
        x_353 = layer_norm_34 = None
        x_360 = x_359.reshape(1, 14, 14, 384)
        x_359 = None
        x_361 = torch.roll(x_360, shifts=(-3, -3), dims=(1, 2))
        x_362 = torch._C._nn.pad(x_361, (0, 0, 0, 0, 0, 0), "constant", None)
        x_361 = None
        x_363 = x_362.view(1, 2, 7, 2, 7, 384)
        x_362 = None
        permute_54 = x_363.permute(0, 1, 3, 2, 4, 5)
        x_363 = None
        contiguous_45 = permute_54.contiguous()
        permute_54 = None
        windows_15 = contiguous_45.view(-1, 7, 7, 384)
        contiguous_45 = None
        x_windows_15 = windows_15.view(-1, 49, 384)
        windows_15 = None
        linear_92 = torch._C._nn.linear(
            x_windows_15,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_,
        )
        x_windows_15 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_122 = linear_92.view(4, 49, 3, 12, 32)
        linear_92 = None
        qkv_15 = view_122.permute(2, 0, 3, 1, 4)
        view_122 = None
        unbind_15 = qkv_15.unbind(0)
        qkv_15 = None
        query_15 = unbind_15[0]
        key_15 = unbind_15[1]
        value_15 = unbind_15[2]
        unbind_15 = None
        normalize_30 = torch.nn.functional.normalize(query_15, dim=-1)
        query_15 = None
        normalize_31 = torch.nn.functional.normalize(key_15, dim=-1)
        key_15 = None
        transpose_45 = normalize_31.transpose(-2, -1)
        normalize_31 = None
        attn_96 = normalize_30 @ transpose_45
        normalize_30 = transpose_45 = None
        reshape_77 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_parameters_logit_scale_.reshape(
            1, 12, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_parameters_logit_scale_ = (
            None
        )
        clamp_15 = torch.clamp(reshape_77, max=4.605170185988092)
        reshape_77 = None
        logit_scale_15 = clamp_15.exp()
        clamp_15 = None
        attn_97 = attn_96 * logit_scale_15
        attn_96 = logit_scale_15 = None
        x_364 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_365 = torch.nn.functional.relu(x_364, inplace=False)
        x_364 = None
        x_366 = torch.nn.functional.dropout(x_365, 0.125, False, False)
        x_365 = None
        x_367 = torch._C._nn.linear(
            x_366,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_366 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_368 = torch.nn.functional.dropout(x_367, 0.0, False, False)
        x_367 = None
        transpose_46 = x_368.transpose(1, 0)
        x_368 = None
        relative_position_bias_30 = transpose_46.reshape(12, 49, 49)
        transpose_46 = None
        relative_position_bias_31 = relative_position_bias_30.unsqueeze(0)
        relative_position_bias_30 = None
        attn_98 = attn_97 + relative_position_bias_31
        attn_97 = relative_position_bias_31 = None
        attn_99 = attn_98.view(1, 4, 12, 49, 49)
        attn_98 = None
        unsqueeze_30 = l_self_modules_stages_modules_2_modules_blocks_modules_11_buffers_attn_mask_.unsqueeze(
            1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_11_buffers_attn_mask_ = (
            None
        )
        unsqueeze_31 = unsqueeze_30.unsqueeze(0)
        unsqueeze_30 = None
        attn_100 = attn_99 + unsqueeze_31
        attn_99 = unsqueeze_31 = None
        attn_101 = attn_100.view(-1, 12, 49, 49)
        attn_100 = None
        attn_102 = attn_101.softmax(dim=-1)
        attn_101 = None
        attn_103 = torch.nn.functional.dropout(attn_102, 0.0, False, False)
        attn_102 = None
        matmul_31 = attn_103 @ value_15
        attn_103 = value_15 = None
        transpose_47 = matmul_31.transpose(1, 2)
        matmul_31 = None
        x_369 = transpose_47.reshape(4, 49, -1)
        transpose_47 = None
        x_370 = torch._C._nn.linear(
            x_369,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_,
        )
        x_369 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_ = (None)
        x_371 = torch.nn.functional.dropout(x_370, 0.0, False, False)
        x_370 = None
        attn_windows_15 = x_371.view(-1, 7, 7, 384)
        x_371 = None
        x_372 = attn_windows_15.view(-1, 2, 2, 7, 7, 384)
        attn_windows_15 = None
        permute_56 = x_372.permute(0, 1, 3, 2, 4, 5)
        x_372 = None
        contiguous_46 = permute_56.contiguous()
        permute_56 = None
        x_373 = contiguous_46.view(-1, 14, 14, 384)
        contiguous_46 = None
        getitem_67 = x_373[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_373 = None
        x_374 = getitem_67.contiguous()
        getitem_67 = None
        x_375 = torch.roll(x_374, shifts=(3, 3), dims=(1, 2))
        x_374 = None
        layer_norm_35 = torch.nn.functional.layer_norm(
            x_375,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_375 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_bias_ = (None)
        x_376 = x_360 + layer_norm_35
        x_360 = layer_norm_35 = None
        x_377 = x_376.reshape(1, -1, 384)
        x_376 = None
        x_378 = torch._C._nn.linear(
            x_377,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_379 = torch._C._nn.gelu(x_378, approximate="none")
        x_378 = None
        x_380 = torch.nn.functional.dropout(x_379, 0.0, False, False)
        x_379 = None
        x_381 = torch._C._nn.linear(
            x_380,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_380 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_382 = torch.nn.functional.dropout(x_381, 0.0, False, False)
        x_381 = None
        layer_norm_36 = torch.nn.functional.layer_norm(
            x_382,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_382 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_bias_ = (None)
        x_383 = x_377 + layer_norm_36
        x_377 = layer_norm_36 = None
        x_384 = x_383.reshape(1, 14, 14, 384)
        x_383 = None
        x_385 = torch._C._nn.pad(x_384, (0, 0, 0, 0, 0, 0), "constant", None)
        x_386 = x_385.view(1, 2, 7, 2, 7, 384)
        x_385 = None
        permute_57 = x_386.permute(0, 1, 3, 2, 4, 5)
        x_386 = None
        contiguous_48 = permute_57.contiguous()
        permute_57 = None
        windows_16 = contiguous_48.view(-1, 7, 7, 384)
        contiguous_48 = None
        x_windows_16 = windows_16.view(-1, 49, 384)
        windows_16 = None
        linear_98 = torch._C._nn.linear(
            x_windows_16,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_bias_,
        )
        x_windows_16 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_131 = linear_98.view(4, 49, 3, 12, 32)
        linear_98 = None
        qkv_16 = view_131.permute(2, 0, 3, 1, 4)
        view_131 = None
        unbind_16 = qkv_16.unbind(0)
        qkv_16 = None
        query_16 = unbind_16[0]
        key_16 = unbind_16[1]
        value_16 = unbind_16[2]
        unbind_16 = None
        normalize_32 = torch.nn.functional.normalize(query_16, dim=-1)
        query_16 = None
        normalize_33 = torch.nn.functional.normalize(key_16, dim=-1)
        key_16 = None
        transpose_48 = normalize_33.transpose(-2, -1)
        normalize_33 = None
        attn_104 = normalize_32 @ transpose_48
        normalize_32 = transpose_48 = None
        reshape_82 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_parameters_logit_scale_.reshape(
            1, 12, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_parameters_logit_scale_ = (
            None
        )
        clamp_16 = torch.clamp(reshape_82, max=4.605170185988092)
        reshape_82 = None
        logit_scale_16 = clamp_16.exp()
        clamp_16 = None
        attn_105 = attn_104 * logit_scale_16
        attn_104 = logit_scale_16 = None
        x_387 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_388 = torch.nn.functional.relu(x_387, inplace=False)
        x_387 = None
        x_389 = torch.nn.functional.dropout(x_388, 0.125, False, False)
        x_388 = None
        x_390 = torch._C._nn.linear(
            x_389,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_389 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_391 = torch.nn.functional.dropout(x_390, 0.0, False, False)
        x_390 = None
        transpose_49 = x_391.transpose(1, 0)
        x_391 = None
        relative_position_bias_32 = transpose_49.reshape(12, 49, 49)
        transpose_49 = None
        relative_position_bias_33 = relative_position_bias_32.unsqueeze(0)
        relative_position_bias_32 = None
        attn_106 = attn_105 + relative_position_bias_33
        attn_105 = relative_position_bias_33 = None
        attn_107 = attn_106.softmax(dim=-1)
        attn_106 = None
        attn_108 = torch.nn.functional.dropout(attn_107, 0.0, False, False)
        attn_107 = None
        matmul_33 = attn_108 @ value_16
        attn_108 = value_16 = None
        transpose_50 = matmul_33.transpose(1, 2)
        matmul_33 = None
        x_392 = transpose_50.reshape(4, 49, -1)
        transpose_50 = None
        x_393 = torch._C._nn.linear(
            x_392,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_,
        )
        x_392 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_ = (None)
        x_394 = torch.nn.functional.dropout(x_393, 0.0, False, False)
        x_393 = None
        attn_windows_16 = x_394.view(-1, 7, 7, 384)
        x_394 = None
        x_395 = attn_windows_16.view(-1, 2, 2, 7, 7, 384)
        attn_windows_16 = None
        permute_59 = x_395.permute(0, 1, 3, 2, 4, 5)
        x_395 = None
        contiguous_49 = permute_59.contiguous()
        permute_59 = None
        x_396 = contiguous_49.view(-1, 14, 14, 384)
        contiguous_49 = None
        getitem_71 = x_396[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_396 = None
        x_397 = getitem_71.contiguous()
        getitem_71 = None
        layer_norm_37 = torch.nn.functional.layer_norm(
            x_397,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_397 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_bias_ = (None)
        x_398 = x_384 + layer_norm_37
        x_384 = layer_norm_37 = None
        x_399 = x_398.reshape(1, -1, 384)
        x_398 = None
        x_400 = torch._C._nn.linear(
            x_399,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_401 = torch._C._nn.gelu(x_400, approximate="none")
        x_400 = None
        x_402 = torch.nn.functional.dropout(x_401, 0.0, False, False)
        x_401 = None
        x_403 = torch._C._nn.linear(
            x_402,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_402 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_404 = torch.nn.functional.dropout(x_403, 0.0, False, False)
        x_403 = None
        layer_norm_38 = torch.nn.functional.layer_norm(
            x_404,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_404 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_bias_ = (None)
        x_405 = x_399 + layer_norm_38
        x_399 = layer_norm_38 = None
        x_406 = x_405.reshape(1, 14, 14, 384)
        x_405 = None
        x_407 = torch.roll(x_406, shifts=(-3, -3), dims=(1, 2))
        x_408 = torch._C._nn.pad(x_407, (0, 0, 0, 0, 0, 0), "constant", None)
        x_407 = None
        x_409 = x_408.view(1, 2, 7, 2, 7, 384)
        x_408 = None
        permute_60 = x_409.permute(0, 1, 3, 2, 4, 5)
        x_409 = None
        contiguous_51 = permute_60.contiguous()
        permute_60 = None
        windows_17 = contiguous_51.view(-1, 7, 7, 384)
        contiguous_51 = None
        x_windows_17 = windows_17.view(-1, 49, 384)
        windows_17 = None
        linear_104 = torch._C._nn.linear(
            x_windows_17,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_bias_,
        )
        x_windows_17 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_138 = linear_104.view(4, 49, 3, 12, 32)
        linear_104 = None
        qkv_17 = view_138.permute(2, 0, 3, 1, 4)
        view_138 = None
        unbind_17 = qkv_17.unbind(0)
        qkv_17 = None
        query_17 = unbind_17[0]
        key_17 = unbind_17[1]
        value_17 = unbind_17[2]
        unbind_17 = None
        normalize_34 = torch.nn.functional.normalize(query_17, dim=-1)
        query_17 = None
        normalize_35 = torch.nn.functional.normalize(key_17, dim=-1)
        key_17 = None
        transpose_51 = normalize_35.transpose(-2, -1)
        normalize_35 = None
        attn_109 = normalize_34 @ transpose_51
        normalize_34 = transpose_51 = None
        reshape_87 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_parameters_logit_scale_.reshape(
            1, 12, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_parameters_logit_scale_ = (
            None
        )
        clamp_17 = torch.clamp(reshape_87, max=4.605170185988092)
        reshape_87 = None
        logit_scale_17 = clamp_17.exp()
        clamp_17 = None
        attn_110 = attn_109 * logit_scale_17
        attn_109 = logit_scale_17 = None
        x_410 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_411 = torch.nn.functional.relu(x_410, inplace=False)
        x_410 = None
        x_412 = torch.nn.functional.dropout(x_411, 0.125, False, False)
        x_411 = None
        x_413 = torch._C._nn.linear(
            x_412,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_412 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_414 = torch.nn.functional.dropout(x_413, 0.0, False, False)
        x_413 = None
        transpose_52 = x_414.transpose(1, 0)
        x_414 = None
        relative_position_bias_34 = transpose_52.reshape(12, 49, 49)
        transpose_52 = None
        relative_position_bias_35 = relative_position_bias_34.unsqueeze(0)
        relative_position_bias_34 = None
        attn_111 = attn_110 + relative_position_bias_35
        attn_110 = relative_position_bias_35 = None
        attn_112 = attn_111.view(1, 4, 12, 49, 49)
        attn_111 = None
        unsqueeze_34 = l_self_modules_stages_modules_2_modules_blocks_modules_13_buffers_attn_mask_.unsqueeze(
            1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_13_buffers_attn_mask_ = (
            None
        )
        unsqueeze_35 = unsqueeze_34.unsqueeze(0)
        unsqueeze_34 = None
        attn_113 = attn_112 + unsqueeze_35
        attn_112 = unsqueeze_35 = None
        attn_114 = attn_113.view(-1, 12, 49, 49)
        attn_113 = None
        attn_115 = attn_114.softmax(dim=-1)
        attn_114 = None
        attn_116 = torch.nn.functional.dropout(attn_115, 0.0, False, False)
        attn_115 = None
        matmul_35 = attn_116 @ value_17
        attn_116 = value_17 = None
        transpose_53 = matmul_35.transpose(1, 2)
        matmul_35 = None
        x_415 = transpose_53.reshape(4, 49, -1)
        transpose_53 = None
        x_416 = torch._C._nn.linear(
            x_415,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_,
        )
        x_415 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_ = (None)
        x_417 = torch.nn.functional.dropout(x_416, 0.0, False, False)
        x_416 = None
        attn_windows_17 = x_417.view(-1, 7, 7, 384)
        x_417 = None
        x_418 = attn_windows_17.view(-1, 2, 2, 7, 7, 384)
        attn_windows_17 = None
        permute_62 = x_418.permute(0, 1, 3, 2, 4, 5)
        x_418 = None
        contiguous_52 = permute_62.contiguous()
        permute_62 = None
        x_419 = contiguous_52.view(-1, 14, 14, 384)
        contiguous_52 = None
        getitem_75 = x_419[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_419 = None
        x_420 = getitem_75.contiguous()
        getitem_75 = None
        x_421 = torch.roll(x_420, shifts=(3, 3), dims=(1, 2))
        x_420 = None
        layer_norm_39 = torch.nn.functional.layer_norm(
            x_421,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_421 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_bias_ = (None)
        x_422 = x_406 + layer_norm_39
        x_406 = layer_norm_39 = None
        x_423 = x_422.reshape(1, -1, 384)
        x_422 = None
        x_424 = torch._C._nn.linear(
            x_423,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_425 = torch._C._nn.gelu(x_424, approximate="none")
        x_424 = None
        x_426 = torch.nn.functional.dropout(x_425, 0.0, False, False)
        x_425 = None
        x_427 = torch._C._nn.linear(
            x_426,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_426 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_428 = torch.nn.functional.dropout(x_427, 0.0, False, False)
        x_427 = None
        layer_norm_40 = torch.nn.functional.layer_norm(
            x_428,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_428 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_bias_ = (None)
        x_429 = x_423 + layer_norm_40
        x_423 = layer_norm_40 = None
        x_430 = x_429.reshape(1, 14, 14, 384)
        x_429 = None
        x_431 = torch._C._nn.pad(x_430, (0, 0, 0, 0, 0, 0), "constant", None)
        x_432 = x_431.view(1, 2, 7, 2, 7, 384)
        x_431 = None
        permute_63 = x_432.permute(0, 1, 3, 2, 4, 5)
        x_432 = None
        contiguous_54 = permute_63.contiguous()
        permute_63 = None
        windows_18 = contiguous_54.view(-1, 7, 7, 384)
        contiguous_54 = None
        x_windows_18 = windows_18.view(-1, 49, 384)
        windows_18 = None
        linear_110 = torch._C._nn.linear(
            x_windows_18,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_bias_,
        )
        x_windows_18 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_147 = linear_110.view(4, 49, 3, 12, 32)
        linear_110 = None
        qkv_18 = view_147.permute(2, 0, 3, 1, 4)
        view_147 = None
        unbind_18 = qkv_18.unbind(0)
        qkv_18 = None
        query_18 = unbind_18[0]
        key_18 = unbind_18[1]
        value_18 = unbind_18[2]
        unbind_18 = None
        normalize_36 = torch.nn.functional.normalize(query_18, dim=-1)
        query_18 = None
        normalize_37 = torch.nn.functional.normalize(key_18, dim=-1)
        key_18 = None
        transpose_54 = normalize_37.transpose(-2, -1)
        normalize_37 = None
        attn_117 = normalize_36 @ transpose_54
        normalize_36 = transpose_54 = None
        reshape_92 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_parameters_logit_scale_.reshape(
            1, 12, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_parameters_logit_scale_ = (
            None
        )
        clamp_18 = torch.clamp(reshape_92, max=4.605170185988092)
        reshape_92 = None
        logit_scale_18 = clamp_18.exp()
        clamp_18 = None
        attn_118 = attn_117 * logit_scale_18
        attn_117 = logit_scale_18 = None
        x_433 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_434 = torch.nn.functional.relu(x_433, inplace=False)
        x_433 = None
        x_435 = torch.nn.functional.dropout(x_434, 0.125, False, False)
        x_434 = None
        x_436 = torch._C._nn.linear(
            x_435,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_435 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_437 = torch.nn.functional.dropout(x_436, 0.0, False, False)
        x_436 = None
        transpose_55 = x_437.transpose(1, 0)
        x_437 = None
        relative_position_bias_36 = transpose_55.reshape(12, 49, 49)
        transpose_55 = None
        relative_position_bias_37 = relative_position_bias_36.unsqueeze(0)
        relative_position_bias_36 = None
        attn_119 = attn_118 + relative_position_bias_37
        attn_118 = relative_position_bias_37 = None
        attn_120 = attn_119.softmax(dim=-1)
        attn_119 = None
        attn_121 = torch.nn.functional.dropout(attn_120, 0.0, False, False)
        attn_120 = None
        matmul_37 = attn_121 @ value_18
        attn_121 = value_18 = None
        transpose_56 = matmul_37.transpose(1, 2)
        matmul_37 = None
        x_438 = transpose_56.reshape(4, 49, -1)
        transpose_56 = None
        x_439 = torch._C._nn.linear(
            x_438,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_,
        )
        x_438 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_ = (None)
        x_440 = torch.nn.functional.dropout(x_439, 0.0, False, False)
        x_439 = None
        attn_windows_18 = x_440.view(-1, 7, 7, 384)
        x_440 = None
        x_441 = attn_windows_18.view(-1, 2, 2, 7, 7, 384)
        attn_windows_18 = None
        permute_65 = x_441.permute(0, 1, 3, 2, 4, 5)
        x_441 = None
        contiguous_55 = permute_65.contiguous()
        permute_65 = None
        x_442 = contiguous_55.view(-1, 14, 14, 384)
        contiguous_55 = None
        getitem_79 = x_442[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_442 = None
        x_443 = getitem_79.contiguous()
        getitem_79 = None
        layer_norm_41 = torch.nn.functional.layer_norm(
            x_443,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_443 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_bias_ = (None)
        x_444 = x_430 + layer_norm_41
        x_430 = layer_norm_41 = None
        x_445 = x_444.reshape(1, -1, 384)
        x_444 = None
        x_446 = torch._C._nn.linear(
            x_445,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_447 = torch._C._nn.gelu(x_446, approximate="none")
        x_446 = None
        x_448 = torch.nn.functional.dropout(x_447, 0.0, False, False)
        x_447 = None
        x_449 = torch._C._nn.linear(
            x_448,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_448 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_450 = torch.nn.functional.dropout(x_449, 0.0, False, False)
        x_449 = None
        layer_norm_42 = torch.nn.functional.layer_norm(
            x_450,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_450 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_bias_ = (None)
        x_451 = x_445 + layer_norm_42
        x_445 = layer_norm_42 = None
        x_452 = x_451.reshape(1, 14, 14, 384)
        x_451 = None
        x_453 = torch.roll(x_452, shifts=(-3, -3), dims=(1, 2))
        x_454 = torch._C._nn.pad(x_453, (0, 0, 0, 0, 0, 0), "constant", None)
        x_453 = None
        x_455 = x_454.view(1, 2, 7, 2, 7, 384)
        x_454 = None
        permute_66 = x_455.permute(0, 1, 3, 2, 4, 5)
        x_455 = None
        contiguous_57 = permute_66.contiguous()
        permute_66 = None
        windows_19 = contiguous_57.view(-1, 7, 7, 384)
        contiguous_57 = None
        x_windows_19 = windows_19.view(-1, 49, 384)
        windows_19 = None
        linear_116 = torch._C._nn.linear(
            x_windows_19,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_bias_,
        )
        x_windows_19 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_154 = linear_116.view(4, 49, 3, 12, 32)
        linear_116 = None
        qkv_19 = view_154.permute(2, 0, 3, 1, 4)
        view_154 = None
        unbind_19 = qkv_19.unbind(0)
        qkv_19 = None
        query_19 = unbind_19[0]
        key_19 = unbind_19[1]
        value_19 = unbind_19[2]
        unbind_19 = None
        normalize_38 = torch.nn.functional.normalize(query_19, dim=-1)
        query_19 = None
        normalize_39 = torch.nn.functional.normalize(key_19, dim=-1)
        key_19 = None
        transpose_57 = normalize_39.transpose(-2, -1)
        normalize_39 = None
        attn_122 = normalize_38 @ transpose_57
        normalize_38 = transpose_57 = None
        reshape_97 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_parameters_logit_scale_.reshape(
            1, 12, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_parameters_logit_scale_ = (
            None
        )
        clamp_19 = torch.clamp(reshape_97, max=4.605170185988092)
        reshape_97 = None
        logit_scale_19 = clamp_19.exp()
        clamp_19 = None
        attn_123 = attn_122 * logit_scale_19
        attn_122 = logit_scale_19 = None
        x_456 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_457 = torch.nn.functional.relu(x_456, inplace=False)
        x_456 = None
        x_458 = torch.nn.functional.dropout(x_457, 0.125, False, False)
        x_457 = None
        x_459 = torch._C._nn.linear(
            x_458,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_458 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_460 = torch.nn.functional.dropout(x_459, 0.0, False, False)
        x_459 = None
        transpose_58 = x_460.transpose(1, 0)
        x_460 = None
        relative_position_bias_38 = transpose_58.reshape(12, 49, 49)
        transpose_58 = None
        relative_position_bias_39 = relative_position_bias_38.unsqueeze(0)
        relative_position_bias_38 = None
        attn_124 = attn_123 + relative_position_bias_39
        attn_123 = relative_position_bias_39 = None
        attn_125 = attn_124.view(1, 4, 12, 49, 49)
        attn_124 = None
        unsqueeze_38 = l_self_modules_stages_modules_2_modules_blocks_modules_15_buffers_attn_mask_.unsqueeze(
            1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_15_buffers_attn_mask_ = (
            None
        )
        unsqueeze_39 = unsqueeze_38.unsqueeze(0)
        unsqueeze_38 = None
        attn_126 = attn_125 + unsqueeze_39
        attn_125 = unsqueeze_39 = None
        attn_127 = attn_126.view(-1, 12, 49, 49)
        attn_126 = None
        attn_128 = attn_127.softmax(dim=-1)
        attn_127 = None
        attn_129 = torch.nn.functional.dropout(attn_128, 0.0, False, False)
        attn_128 = None
        matmul_39 = attn_129 @ value_19
        attn_129 = value_19 = None
        transpose_59 = matmul_39.transpose(1, 2)
        matmul_39 = None
        x_461 = transpose_59.reshape(4, 49, -1)
        transpose_59 = None
        x_462 = torch._C._nn.linear(
            x_461,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_,
        )
        x_461 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_ = (None)
        x_463 = torch.nn.functional.dropout(x_462, 0.0, False, False)
        x_462 = None
        attn_windows_19 = x_463.view(-1, 7, 7, 384)
        x_463 = None
        x_464 = attn_windows_19.view(-1, 2, 2, 7, 7, 384)
        attn_windows_19 = None
        permute_68 = x_464.permute(0, 1, 3, 2, 4, 5)
        x_464 = None
        contiguous_58 = permute_68.contiguous()
        permute_68 = None
        x_465 = contiguous_58.view(-1, 14, 14, 384)
        contiguous_58 = None
        getitem_83 = x_465[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_465 = None
        x_466 = getitem_83.contiguous()
        getitem_83 = None
        x_467 = torch.roll(x_466, shifts=(3, 3), dims=(1, 2))
        x_466 = None
        layer_norm_43 = torch.nn.functional.layer_norm(
            x_467,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_467 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_bias_ = (None)
        x_468 = x_452 + layer_norm_43
        x_452 = layer_norm_43 = None
        x_469 = x_468.reshape(1, -1, 384)
        x_468 = None
        x_470 = torch._C._nn.linear(
            x_469,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_471 = torch._C._nn.gelu(x_470, approximate="none")
        x_470 = None
        x_472 = torch.nn.functional.dropout(x_471, 0.0, False, False)
        x_471 = None
        x_473 = torch._C._nn.linear(
            x_472,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_472 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_474 = torch.nn.functional.dropout(x_473, 0.0, False, False)
        x_473 = None
        layer_norm_44 = torch.nn.functional.layer_norm(
            x_474,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_474 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_bias_ = (None)
        x_475 = x_469 + layer_norm_44
        x_469 = layer_norm_44 = None
        x_476 = x_475.reshape(1, 14, 14, 384)
        x_475 = None
        x_477 = torch._C._nn.pad(x_476, (0, 0, 0, 0, 0, 0), "constant", None)
        x_478 = x_477.view(1, 2, 7, 2, 7, 384)
        x_477 = None
        permute_69 = x_478.permute(0, 1, 3, 2, 4, 5)
        x_478 = None
        contiguous_60 = permute_69.contiguous()
        permute_69 = None
        windows_20 = contiguous_60.view(-1, 7, 7, 384)
        contiguous_60 = None
        x_windows_20 = windows_20.view(-1, 49, 384)
        windows_20 = None
        linear_122 = torch._C._nn.linear(
            x_windows_20,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_bias_,
        )
        x_windows_20 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_163 = linear_122.view(4, 49, 3, 12, 32)
        linear_122 = None
        qkv_20 = view_163.permute(2, 0, 3, 1, 4)
        view_163 = None
        unbind_20 = qkv_20.unbind(0)
        qkv_20 = None
        query_20 = unbind_20[0]
        key_20 = unbind_20[1]
        value_20 = unbind_20[2]
        unbind_20 = None
        normalize_40 = torch.nn.functional.normalize(query_20, dim=-1)
        query_20 = None
        normalize_41 = torch.nn.functional.normalize(key_20, dim=-1)
        key_20 = None
        transpose_60 = normalize_41.transpose(-2, -1)
        normalize_41 = None
        attn_130 = normalize_40 @ transpose_60
        normalize_40 = transpose_60 = None
        reshape_102 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_parameters_logit_scale_.reshape(
            1, 12, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_parameters_logit_scale_ = (
            None
        )
        clamp_20 = torch.clamp(reshape_102, max=4.605170185988092)
        reshape_102 = None
        logit_scale_20 = clamp_20.exp()
        clamp_20 = None
        attn_131 = attn_130 * logit_scale_20
        attn_130 = logit_scale_20 = None
        x_479 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_480 = torch.nn.functional.relu(x_479, inplace=False)
        x_479 = None
        x_481 = torch.nn.functional.dropout(x_480, 0.125, False, False)
        x_480 = None
        x_482 = torch._C._nn.linear(
            x_481,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_481 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_483 = torch.nn.functional.dropout(x_482, 0.0, False, False)
        x_482 = None
        transpose_61 = x_483.transpose(1, 0)
        x_483 = None
        relative_position_bias_40 = transpose_61.reshape(12, 49, 49)
        transpose_61 = None
        relative_position_bias_41 = relative_position_bias_40.unsqueeze(0)
        relative_position_bias_40 = None
        attn_132 = attn_131 + relative_position_bias_41
        attn_131 = relative_position_bias_41 = None
        attn_133 = attn_132.softmax(dim=-1)
        attn_132 = None
        attn_134 = torch.nn.functional.dropout(attn_133, 0.0, False, False)
        attn_133 = None
        matmul_41 = attn_134 @ value_20
        attn_134 = value_20 = None
        transpose_62 = matmul_41.transpose(1, 2)
        matmul_41 = None
        x_484 = transpose_62.reshape(4, 49, -1)
        transpose_62 = None
        x_485 = torch._C._nn.linear(
            x_484,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_,
        )
        x_484 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_ = (None)
        x_486 = torch.nn.functional.dropout(x_485, 0.0, False, False)
        x_485 = None
        attn_windows_20 = x_486.view(-1, 7, 7, 384)
        x_486 = None
        x_487 = attn_windows_20.view(-1, 2, 2, 7, 7, 384)
        attn_windows_20 = None
        permute_71 = x_487.permute(0, 1, 3, 2, 4, 5)
        x_487 = None
        contiguous_61 = permute_71.contiguous()
        permute_71 = None
        x_488 = contiguous_61.view(-1, 14, 14, 384)
        contiguous_61 = None
        getitem_87 = x_488[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_488 = None
        x_489 = getitem_87.contiguous()
        getitem_87 = None
        layer_norm_45 = torch.nn.functional.layer_norm(
            x_489,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_489 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_bias_ = (None)
        x_490 = x_476 + layer_norm_45
        x_476 = layer_norm_45 = None
        x_491 = x_490.reshape(1, -1, 384)
        x_490 = None
        x_492 = torch._C._nn.linear(
            x_491,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_493 = torch._C._nn.gelu(x_492, approximate="none")
        x_492 = None
        x_494 = torch.nn.functional.dropout(x_493, 0.0, False, False)
        x_493 = None
        x_495 = torch._C._nn.linear(
            x_494,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_494 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_496 = torch.nn.functional.dropout(x_495, 0.0, False, False)
        x_495 = None
        layer_norm_46 = torch.nn.functional.layer_norm(
            x_496,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_496 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_bias_ = (None)
        x_497 = x_491 + layer_norm_46
        x_491 = layer_norm_46 = None
        x_498 = x_497.reshape(1, 14, 14, 384)
        x_497 = None
        x_499 = torch.roll(x_498, shifts=(-3, -3), dims=(1, 2))
        x_500 = torch._C._nn.pad(x_499, (0, 0, 0, 0, 0, 0), "constant", None)
        x_499 = None
        x_501 = x_500.view(1, 2, 7, 2, 7, 384)
        x_500 = None
        permute_72 = x_501.permute(0, 1, 3, 2, 4, 5)
        x_501 = None
        contiguous_63 = permute_72.contiguous()
        permute_72 = None
        windows_21 = contiguous_63.view(-1, 7, 7, 384)
        contiguous_63 = None
        x_windows_21 = windows_21.view(-1, 49, 384)
        windows_21 = None
        linear_128 = torch._C._nn.linear(
            x_windows_21,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_bias_,
        )
        x_windows_21 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_170 = linear_128.view(4, 49, 3, 12, 32)
        linear_128 = None
        qkv_21 = view_170.permute(2, 0, 3, 1, 4)
        view_170 = None
        unbind_21 = qkv_21.unbind(0)
        qkv_21 = None
        query_21 = unbind_21[0]
        key_21 = unbind_21[1]
        value_21 = unbind_21[2]
        unbind_21 = None
        normalize_42 = torch.nn.functional.normalize(query_21, dim=-1)
        query_21 = None
        normalize_43 = torch.nn.functional.normalize(key_21, dim=-1)
        key_21 = None
        transpose_63 = normalize_43.transpose(-2, -1)
        normalize_43 = None
        attn_135 = normalize_42 @ transpose_63
        normalize_42 = transpose_63 = None
        reshape_107 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_parameters_logit_scale_.reshape(
            1, 12, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_parameters_logit_scale_ = (
            None
        )
        clamp_21 = torch.clamp(reshape_107, max=4.605170185988092)
        reshape_107 = None
        logit_scale_21 = clamp_21.exp()
        clamp_21 = None
        attn_136 = attn_135 * logit_scale_21
        attn_135 = logit_scale_21 = None
        x_502 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_503 = torch.nn.functional.relu(x_502, inplace=False)
        x_502 = None
        x_504 = torch.nn.functional.dropout(x_503, 0.125, False, False)
        x_503 = None
        x_505 = torch._C._nn.linear(
            x_504,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_504 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_506 = torch.nn.functional.dropout(x_505, 0.0, False, False)
        x_505 = None
        transpose_64 = x_506.transpose(1, 0)
        x_506 = None
        relative_position_bias_42 = transpose_64.reshape(12, 49, 49)
        transpose_64 = None
        relative_position_bias_43 = relative_position_bias_42.unsqueeze(0)
        relative_position_bias_42 = None
        attn_137 = attn_136 + relative_position_bias_43
        attn_136 = relative_position_bias_43 = None
        attn_138 = attn_137.view(1, 4, 12, 49, 49)
        attn_137 = None
        unsqueeze_42 = l_self_modules_stages_modules_2_modules_blocks_modules_17_buffers_attn_mask_.unsqueeze(
            1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_17_buffers_attn_mask_ = (
            None
        )
        unsqueeze_43 = unsqueeze_42.unsqueeze(0)
        unsqueeze_42 = None
        attn_139 = attn_138 + unsqueeze_43
        attn_138 = unsqueeze_43 = None
        attn_140 = attn_139.view(-1, 12, 49, 49)
        attn_139 = None
        attn_141 = attn_140.softmax(dim=-1)
        attn_140 = None
        attn_142 = torch.nn.functional.dropout(attn_141, 0.0, False, False)
        attn_141 = None
        matmul_43 = attn_142 @ value_21
        attn_142 = value_21 = None
        transpose_65 = matmul_43.transpose(1, 2)
        matmul_43 = None
        x_507 = transpose_65.reshape(4, 49, -1)
        transpose_65 = None
        x_508 = torch._C._nn.linear(
            x_507,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_,
        )
        x_507 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_ = (None)
        x_509 = torch.nn.functional.dropout(x_508, 0.0, False, False)
        x_508 = None
        attn_windows_21 = x_509.view(-1, 7, 7, 384)
        x_509 = None
        x_510 = attn_windows_21.view(-1, 2, 2, 7, 7, 384)
        attn_windows_21 = None
        permute_74 = x_510.permute(0, 1, 3, 2, 4, 5)
        x_510 = None
        contiguous_64 = permute_74.contiguous()
        permute_74 = None
        x_511 = contiguous_64.view(-1, 14, 14, 384)
        contiguous_64 = None
        getitem_91 = x_511[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_511 = None
        x_512 = getitem_91.contiguous()
        getitem_91 = None
        x_513 = torch.roll(x_512, shifts=(3, 3), dims=(1, 2))
        x_512 = None
        layer_norm_47 = torch.nn.functional.layer_norm(
            x_513,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_513 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_bias_ = (None)
        x_514 = x_498 + layer_norm_47
        x_498 = layer_norm_47 = None
        x_515 = x_514.reshape(1, -1, 384)
        x_514 = None
        x_516 = torch._C._nn.linear(
            x_515,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_517 = torch._C._nn.gelu(x_516, approximate="none")
        x_516 = None
        x_518 = torch.nn.functional.dropout(x_517, 0.0, False, False)
        x_517 = None
        x_519 = torch._C._nn.linear(
            x_518,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_518 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_520 = torch.nn.functional.dropout(x_519, 0.0, False, False)
        x_519 = None
        layer_norm_48 = torch.nn.functional.layer_norm(
            x_520,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_520 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_bias_ = (None)
        x_521 = x_515 + layer_norm_48
        x_515 = layer_norm_48 = None
        x_522 = torch.nn.functional.layer_norm(
            x_521,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm3_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm3_parameters_bias_,
            1e-05,
        )
        x_521 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm3_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm3_parameters_bias_ = (None)
        x_523 = x_522.reshape(1, 14, 14, 384)
        x_522 = None
        x_524 = x_523.permute(0, 3, 1, 2)
        x_523 = None
        x_525 = x_524.permute(0, 2, 3, 1)
        x_524 = None
        x_526 = torch._C._nn.pad(x_525, (0, 0, 0, 0, 0, 0), "constant", None)
        x_525 = None
        reshape_112 = x_526.reshape(1, 7, 2, 7, 2, 384)
        x_526 = None
        permute_77 = reshape_112.permute(0, 1, 3, 4, 2, 5)
        reshape_112 = None
        x_527 = permute_77.flatten(3)
        permute_77 = None
        x_528 = torch.nn.functional.layer_norm(
            x_527,
            (1536,),
            l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_bias_,
            1e-05,
        )
        x_527 = l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_ = l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_bias_ = (None)
        x_529 = torch._C._nn.linear(
            x_528,
            l_self_modules_stages_modules_3_modules_downsample_modules_reduction_parameters_weight_,
            None,
        )
        x_528 = l_self_modules_stages_modules_3_modules_downsample_modules_reduction_parameters_weight_ = (None)
        x_530 = torch._C._nn.pad(x_529, (0, 0, 0, 0, 0, 0), "constant", None)
        x_531 = x_530.view(1, 1, 7, 1, 7, 768)
        x_530 = None
        permute_78 = x_531.permute(0, 1, 3, 2, 4, 5)
        x_531 = None
        contiguous_66 = permute_78.contiguous()
        permute_78 = None
        windows_22 = contiguous_66.view(-1, 7, 7, 768)
        contiguous_66 = None
        x_windows_22 = windows_22.view(-1, 49, 768)
        windows_22 = None
        linear_135 = torch._C._nn.linear(
            x_windows_22,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        x_windows_22 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_179 = linear_135.view(1, 49, 3, 24, 32)
        linear_135 = None
        qkv_22 = view_179.permute(2, 0, 3, 1, 4)
        view_179 = None
        unbind_22 = qkv_22.unbind(0)
        qkv_22 = None
        query_22 = unbind_22[0]
        key_22 = unbind_22[1]
        value_22 = unbind_22[2]
        unbind_22 = None
        normalize_44 = torch.nn.functional.normalize(query_22, dim=-1)
        query_22 = None
        normalize_45 = torch.nn.functional.normalize(key_22, dim=-1)
        key_22 = None
        transpose_66 = normalize_45.transpose(-2, -1)
        normalize_45 = None
        attn_143 = normalize_44 @ transpose_66
        normalize_44 = transpose_66 = None
        reshape_113 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_parameters_logit_scale_.reshape(
            1, 24, 1, 1
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_parameters_logit_scale_ = (
            None
        )
        clamp_22 = torch.clamp(reshape_113, max=4.605170185988092)
        reshape_113 = None
        logit_scale_22 = clamp_22.exp()
        clamp_22 = None
        attn_144 = attn_143 * logit_scale_22
        attn_143 = logit_scale_22 = None
        x_532 = torch._C._nn.linear(
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_533 = torch.nn.functional.relu(x_532, inplace=False)
        x_532 = None
        x_534 = torch.nn.functional.dropout(x_533, 0.125, False, False)
        x_533 = None
        x_535 = torch._C._nn.linear(
            x_534,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_534 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_536 = torch.nn.functional.dropout(x_535, 0.0, False, False)
        x_535 = None
        transpose_67 = x_536.transpose(1, 0)
        x_536 = None
        relative_position_bias_44 = transpose_67.reshape(24, 49, 49)
        transpose_67 = None
        relative_position_bias_45 = relative_position_bias_44.unsqueeze(0)
        relative_position_bias_44 = None
        attn_145 = attn_144 + relative_position_bias_45
        attn_144 = relative_position_bias_45 = None
        attn_146 = attn_145.softmax(dim=-1)
        attn_145 = None
        attn_147 = torch.nn.functional.dropout(attn_146, 0.0, False, False)
        attn_146 = None
        matmul_45 = attn_147 @ value_22
        attn_147 = value_22 = None
        transpose_68 = matmul_45.transpose(1, 2)
        matmul_45 = None
        x_537 = transpose_68.reshape(1, 49, -1)
        transpose_68 = None
        x_538 = torch._C._nn.linear(
            x_537,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_537 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_539 = torch.nn.functional.dropout(x_538, 0.0, False, False)
        x_538 = None
        attn_windows_22 = x_539.view(-1, 7, 7, 768)
        x_539 = None
        x_540 = attn_windows_22.view(-1, 1, 1, 7, 7, 768)
        attn_windows_22 = None
        permute_80 = x_540.permute(0, 1, 3, 2, 4, 5)
        x_540 = None
        contiguous_67 = permute_80.contiguous()
        permute_80 = None
        x_541 = contiguous_67.view(-1, 7, 7, 768)
        contiguous_67 = None
        getitem_95 = x_541[
            (
                slice(None, None, None),
                slice(None, 7, None),
                slice(None, 7, None),
                slice(None, None, None),
            )
        ]
        x_541 = None
        x_542 = getitem_95.contiguous()
        getitem_95 = None
        layer_norm_51 = torch.nn.functional.layer_norm(
            x_542,
            (768,),
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_542 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (None)
        x_543 = x_529 + layer_norm_51
        x_529 = layer_norm_51 = None
        x_544 = x_543.reshape(1, -1, 768)
        x_543 = None
        x_545 = torch._C._nn.linear(
            x_544,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_546 = torch._C._nn.gelu(x_545, approximate="none")
        x_545 = None
        x_547 = torch.nn.functional.dropout(x_546, 0.0, False, False)
        x_546 = None
        x_548 = torch._C._nn.linear(
            x_547,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_547 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_549 = torch.nn.functional.dropout(x_548, 0.0, False, False)
        x_548 = None
        layer_norm_52 = torch.nn.functional.layer_norm(
            x_549,
            (768,),
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_549 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_550 = x_544 + layer_norm_52
        x_544 = layer_norm_52 = None
        x_551 = x_550.reshape(1, 7, 7, 768)
        x_550 = None
        x_552 = torch._C._nn.pad(x_551, (0, 0, 0, 0, 0, 0), "constant", None)
        x_553 = x_552.view(1, 1, 7, 1, 7, 768)
        x_552 = None
        permute_81 = x_553.permute(0, 1, 3, 2, 4, 5)
        x_553 = None
        contiguous_69 = permute_81.contiguous()
        permute_81 = None
        windows_23 = contiguous_69.view(-1, 7, 7, 768)
        contiguous_69 = None
        x_windows_23 = windows_23.view(-1, 49, 768)
        windows_23 = None
        linear_141 = torch._C._nn.linear(
            x_windows_23,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_,
        )
        x_windows_23 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_186 = linear_141.view(1, 49, 3, 24, 32)
        linear_141 = None
        qkv_23 = view_186.permute(2, 0, 3, 1, 4)
        view_186 = None
        unbind_23 = qkv_23.unbind(0)
        qkv_23 = None
        query_23 = unbind_23[0]
        key_23 = unbind_23[1]
        value_23 = unbind_23[2]
        unbind_23 = None
        normalize_46 = torch.nn.functional.normalize(query_23, dim=-1)
        query_23 = None
        normalize_47 = torch.nn.functional.normalize(key_23, dim=-1)
        key_23 = None
        transpose_69 = normalize_47.transpose(-2, -1)
        normalize_47 = None
        attn_148 = normalize_46 @ transpose_69
        normalize_46 = transpose_69 = None
        reshape_118 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_parameters_logit_scale_.reshape(
            1, 24, 1, 1
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_parameters_logit_scale_ = (
            None
        )
        clamp_23 = torch.clamp(reshape_118, max=4.605170185988092)
        reshape_118 = None
        logit_scale_23 = clamp_23.exp()
        clamp_23 = None
        attn_149 = attn_148 * logit_scale_23
        attn_148 = logit_scale_23 = None
        x_554 = torch._C._nn.linear(
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_555 = torch.nn.functional.relu(x_554, inplace=False)
        x_554 = None
        x_556 = torch.nn.functional.dropout(x_555, 0.125, False, False)
        x_555 = None
        x_557 = torch._C._nn.linear(
            x_556,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_556 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_558 = torch.nn.functional.dropout(x_557, 0.0, False, False)
        x_557 = None
        transpose_70 = x_558.transpose(1, 0)
        x_558 = None
        relative_position_bias_46 = transpose_70.reshape(24, 49, 49)
        transpose_70 = None
        relative_position_bias_47 = relative_position_bias_46.unsqueeze(0)
        relative_position_bias_46 = None
        attn_150 = attn_149 + relative_position_bias_47
        attn_149 = relative_position_bias_47 = None
        attn_151 = attn_150.softmax(dim=-1)
        attn_150 = None
        attn_152 = torch.nn.functional.dropout(attn_151, 0.0, False, False)
        attn_151 = None
        matmul_47 = attn_152 @ value_23
        attn_152 = value_23 = None
        transpose_71 = matmul_47.transpose(1, 2)
        matmul_47 = None
        x_559 = transpose_71.reshape(1, 49, -1)
        transpose_71 = None
        x_560 = torch._C._nn.linear(
            x_559,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_559 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_561 = torch.nn.functional.dropout(x_560, 0.0, False, False)
        x_560 = None
        attn_windows_23 = x_561.view(-1, 7, 7, 768)
        x_561 = None
        x_562 = attn_windows_23.view(-1, 1, 1, 7, 7, 768)
        attn_windows_23 = None
        permute_83 = x_562.permute(0, 1, 3, 2, 4, 5)
        x_562 = None
        contiguous_70 = permute_83.contiguous()
        permute_83 = None
        x_563 = contiguous_70.view(-1, 7, 7, 768)
        contiguous_70 = None
        getitem_99 = x_563[
            (
                slice(None, None, None),
                slice(None, 7, None),
                slice(None, 7, None),
                slice(None, None, None),
            )
        ]
        x_563 = None
        x_564 = getitem_99.contiguous()
        getitem_99 = None
        layer_norm_53 = torch.nn.functional.layer_norm(
            x_564,
            (768,),
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_564 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_565 = x_551 + layer_norm_53
        x_551 = layer_norm_53 = None
        x_566 = x_565.reshape(1, -1, 768)
        x_565 = None
        x_567 = torch._C._nn.linear(
            x_566,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_568 = torch._C._nn.gelu(x_567, approximate="none")
        x_567 = None
        x_569 = torch.nn.functional.dropout(x_568, 0.0, False, False)
        x_568 = None
        x_570 = torch._C._nn.linear(
            x_569,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_569 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_571 = torch.nn.functional.dropout(x_570, 0.0, False, False)
        x_570 = None
        layer_norm_54 = torch.nn.functional.layer_norm(
            x_571,
            (768,),
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_571 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_572 = x_566 + layer_norm_54
        x_566 = layer_norm_54 = None
        x_573 = torch.nn.functional.layer_norm(
            x_572,
            (768,),
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_parameters_bias_,
            1e-05,
        )
        x_572 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_parameters_bias_ = (None)
        x_574 = x_573.reshape(1, 7, 7, 768)
        x_573 = None
        x_575 = x_574.permute(0, 3, 1, 2)
        x_574 = None
        x_576 = torch.nn.functional.adaptive_avg_pool2d(x_575, 1)
        x_575 = None
        x_577 = x_576.flatten(1, -1)
        x_576 = None
        x_578 = torch.nn.functional.dropout(x_577, 0.0, False, False)
        x_577 = None
        x_579 = torch._C._nn.linear(
            x_578,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_578 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_579,)
