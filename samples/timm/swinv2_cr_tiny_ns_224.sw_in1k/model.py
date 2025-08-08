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
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm3_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm3_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm3_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm3_parameters_bias_
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
        x_246 = torch.nn.functional.layer_norm(
            x_245,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm3_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm3_parameters_bias_,
            1e-05,
        )
        x_245 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm3_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm3_parameters_bias_ = (None)
        x_247 = x_246.reshape(1, 14, 14, 384)
        x_246 = None
        x_248 = x_247.permute(0, 3, 1, 2)
        x_247 = None
        x_249 = x_248.permute(0, 2, 3, 1)
        x_248 = None
        x_250 = torch._C._nn.pad(x_249, (0, 0, 0, 0, 0, 0), "constant", None)
        x_249 = None
        reshape_52 = x_250.reshape(1, 7, 2, 7, 2, 384)
        x_250 = None
        permute_41 = reshape_52.permute(0, 1, 3, 4, 2, 5)
        reshape_52 = None
        x_251 = permute_41.flatten(3)
        permute_41 = None
        x_252 = torch.nn.functional.layer_norm(
            x_251,
            (1536,),
            l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_bias_,
            1e-05,
        )
        x_251 = l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_ = l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_bias_ = (None)
        x_253 = torch._C._nn.linear(
            x_252,
            l_self_modules_stages_modules_3_modules_downsample_modules_reduction_parameters_weight_,
            None,
        )
        x_252 = l_self_modules_stages_modules_3_modules_downsample_modules_reduction_parameters_weight_ = (None)
        x_254 = torch._C._nn.pad(x_253, (0, 0, 0, 0, 0, 0), "constant", None)
        x_255 = x_254.view(1, 1, 7, 1, 7, 768)
        x_254 = None
        permute_42 = x_255.permute(0, 1, 3, 2, 4, 5)
        x_255 = None
        contiguous_30 = permute_42.contiguous()
        permute_42 = None
        windows_10 = contiguous_30.view(-1, 7, 7, 768)
        contiguous_30 = None
        x_windows_10 = windows_10.view(-1, 49, 768)
        windows_10 = None
        linear_63 = torch._C._nn.linear(
            x_windows_10,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        x_windows_10 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_83 = linear_63.view(1, 49, 3, 24, 32)
        linear_63 = None
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
        reshape_53 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_parameters_logit_scale_.reshape(
            1, 24, 1, 1
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_parameters_logit_scale_ = (
            None
        )
        clamp_10 = torch.clamp(reshape_53, max=4.605170185988092)
        reshape_53 = None
        logit_scale_10 = clamp_10.exp()
        clamp_10 = None
        attn_66 = attn_65 * logit_scale_10
        attn_65 = logit_scale_10 = None
        x_256 = torch._C._nn.linear(
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_257 = torch.nn.functional.relu(x_256, inplace=False)
        x_256 = None
        x_258 = torch.nn.functional.dropout(x_257, 0.125, False, False)
        x_257 = None
        x_259 = torch._C._nn.linear(
            x_258,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_258 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_260 = torch.nn.functional.dropout(x_259, 0.0, False, False)
        x_259 = None
        transpose_31 = x_260.transpose(1, 0)
        x_260 = None
        relative_position_bias_20 = transpose_31.reshape(24, 49, 49)
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
        x_261 = transpose_32.reshape(1, 49, -1)
        transpose_32 = None
        x_262 = torch._C._nn.linear(
            x_261,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_261 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_263 = torch.nn.functional.dropout(x_262, 0.0, False, False)
        x_262 = None
        attn_windows_10 = x_263.view(-1, 7, 7, 768)
        x_263 = None
        x_264 = attn_windows_10.view(-1, 1, 1, 7, 7, 768)
        attn_windows_10 = None
        permute_44 = x_264.permute(0, 1, 3, 2, 4, 5)
        x_264 = None
        contiguous_31 = permute_44.contiguous()
        permute_44 = None
        x_265 = contiguous_31.view(-1, 7, 7, 768)
        contiguous_31 = None
        getitem_47 = x_265[
            (
                slice(None, None, None),
                slice(None, 7, None),
                slice(None, 7, None),
                slice(None, None, None),
            )
        ]
        x_265 = None
        x_266 = getitem_47.contiguous()
        getitem_47 = None
        layer_norm_27 = torch.nn.functional.layer_norm(
            x_266,
            (768,),
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_266 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (None)
        x_267 = x_253 + layer_norm_27
        x_253 = layer_norm_27 = None
        x_268 = x_267.reshape(1, -1, 768)
        x_267 = None
        x_269 = torch._C._nn.linear(
            x_268,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_270 = torch._C._nn.gelu(x_269, approximate="none")
        x_269 = None
        x_271 = torch.nn.functional.dropout(x_270, 0.0, False, False)
        x_270 = None
        x_272 = torch._C._nn.linear(
            x_271,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_271 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_273 = torch.nn.functional.dropout(x_272, 0.0, False, False)
        x_272 = None
        layer_norm_28 = torch.nn.functional.layer_norm(
            x_273,
            (768,),
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_273 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_274 = x_268 + layer_norm_28
        x_268 = layer_norm_28 = None
        x_275 = x_274.reshape(1, 7, 7, 768)
        x_274 = None
        x_276 = torch._C._nn.pad(x_275, (0, 0, 0, 0, 0, 0), "constant", None)
        x_277 = x_276.view(1, 1, 7, 1, 7, 768)
        x_276 = None
        permute_45 = x_277.permute(0, 1, 3, 2, 4, 5)
        x_277 = None
        contiguous_33 = permute_45.contiguous()
        permute_45 = None
        windows_11 = contiguous_33.view(-1, 7, 7, 768)
        contiguous_33 = None
        x_windows_11 = windows_11.view(-1, 49, 768)
        windows_11 = None
        linear_69 = torch._C._nn.linear(
            x_windows_11,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_,
        )
        x_windows_11 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_90 = linear_69.view(1, 49, 3, 24, 32)
        linear_69 = None
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
        reshape_58 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_parameters_logit_scale_.reshape(
            1, 24, 1, 1
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_parameters_logit_scale_ = (
            None
        )
        clamp_11 = torch.clamp(reshape_58, max=4.605170185988092)
        reshape_58 = None
        logit_scale_11 = clamp_11.exp()
        clamp_11 = None
        attn_71 = attn_70 * logit_scale_11
        attn_70 = logit_scale_11 = None
        x_278 = torch._C._nn.linear(
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_279 = torch.nn.functional.relu(x_278, inplace=False)
        x_278 = None
        x_280 = torch.nn.functional.dropout(x_279, 0.125, False, False)
        x_279 = None
        x_281 = torch._C._nn.linear(
            x_280,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_280 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_282 = torch.nn.functional.dropout(x_281, 0.0, False, False)
        x_281 = None
        transpose_34 = x_282.transpose(1, 0)
        x_282 = None
        relative_position_bias_22 = transpose_34.reshape(24, 49, 49)
        transpose_34 = None
        relative_position_bias_23 = relative_position_bias_22.unsqueeze(0)
        relative_position_bias_22 = None
        attn_72 = attn_71 + relative_position_bias_23
        attn_71 = relative_position_bias_23 = None
        attn_73 = attn_72.softmax(dim=-1)
        attn_72 = None
        attn_74 = torch.nn.functional.dropout(attn_73, 0.0, False, False)
        attn_73 = None
        matmul_23 = attn_74 @ value_11
        attn_74 = value_11 = None
        transpose_35 = matmul_23.transpose(1, 2)
        matmul_23 = None
        x_283 = transpose_35.reshape(1, 49, -1)
        transpose_35 = None
        x_284 = torch._C._nn.linear(
            x_283,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_283 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_285 = torch.nn.functional.dropout(x_284, 0.0, False, False)
        x_284 = None
        attn_windows_11 = x_285.view(-1, 7, 7, 768)
        x_285 = None
        x_286 = attn_windows_11.view(-1, 1, 1, 7, 7, 768)
        attn_windows_11 = None
        permute_47 = x_286.permute(0, 1, 3, 2, 4, 5)
        x_286 = None
        contiguous_34 = permute_47.contiguous()
        permute_47 = None
        x_287 = contiguous_34.view(-1, 7, 7, 768)
        contiguous_34 = None
        getitem_51 = x_287[
            (
                slice(None, None, None),
                slice(None, 7, None),
                slice(None, 7, None),
                slice(None, None, None),
            )
        ]
        x_287 = None
        x_288 = getitem_51.contiguous()
        getitem_51 = None
        layer_norm_29 = torch.nn.functional.layer_norm(
            x_288,
            (768,),
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_288 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_289 = x_275 + layer_norm_29
        x_275 = layer_norm_29 = None
        x_290 = x_289.reshape(1, -1, 768)
        x_289 = None
        x_291 = torch._C._nn.linear(
            x_290,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_292 = torch._C._nn.gelu(x_291, approximate="none")
        x_291 = None
        x_293 = torch.nn.functional.dropout(x_292, 0.0, False, False)
        x_292 = None
        x_294 = torch._C._nn.linear(
            x_293,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_293 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_295 = torch.nn.functional.dropout(x_294, 0.0, False, False)
        x_294 = None
        layer_norm_30 = torch.nn.functional.layer_norm(
            x_295,
            (768,),
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_295 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_296 = x_290 + layer_norm_30
        x_290 = layer_norm_30 = None
        x_297 = torch.nn.functional.layer_norm(
            x_296,
            (768,),
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_parameters_bias_,
            1e-05,
        )
        x_296 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_parameters_bias_ = (None)
        x_298 = x_297.reshape(1, 7, 7, 768)
        x_297 = None
        x_299 = x_298.permute(0, 3, 1, 2)
        x_298 = None
        x_300 = torch.nn.functional.adaptive_avg_pool2d(x_299, 1)
        x_299 = None
        x_301 = x_300.flatten(1, -1)
        x_300 = None
        x_302 = torch.nn.functional.dropout(x_301, 0.0, False, False)
        x_301 = None
        x_303 = torch._C._nn.linear(
            x_302,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_302 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_303,)
