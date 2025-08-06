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
        x_48 = x_47.reshape(1, 56, 56, 96)
        x_47 = None
        x_49 = x_48.permute(0, 3, 1, 2)
        x_48 = None
        x_50 = x_49.permute(0, 2, 3, 1)
        x_49 = None
        x_51 = torch._C._nn.pad(x_50, (0, 0, 0, 0, 0, 0), "constant", None)
        x_50 = None
        reshape_10 = x_51.reshape(1, 28, 2, 28, 2, 96)
        x_51 = None
        permute_11 = reshape_10.permute(0, 1, 3, 4, 2, 5)
        reshape_10 = None
        x_52 = permute_11.flatten(3)
        permute_11 = None
        x_53 = torch.nn.functional.layer_norm(
            x_52,
            (384,),
            l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_bias_,
            1e-05,
        )
        x_52 = l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_bias_ = (None)
        x_54 = torch._C._nn.linear(
            x_53,
            l_self_modules_stages_modules_1_modules_downsample_modules_reduction_parameters_weight_,
            None,
        )
        x_53 = l_self_modules_stages_modules_1_modules_downsample_modules_reduction_parameters_weight_ = (None)
        x_55 = torch._C._nn.pad(x_54, (0, 0, 0, 0, 0, 0), "constant", None)
        x_56 = x_55.view(1, 4, 7, 4, 7, 192)
        x_55 = None
        permute_12 = x_56.permute(0, 1, 3, 2, 4, 5)
        x_56 = None
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
        x_57 = torch._C._nn.linear(
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_58 = torch.nn.functional.relu(x_57, inplace=False)
        x_57 = None
        x_59 = torch.nn.functional.dropout(x_58, 0.125, False, False)
        x_58 = None
        x_60 = torch._C._nn.linear(
            x_59,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_59 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_61 = torch.nn.functional.dropout(x_60, 0.0, False, False)
        x_60 = None
        transpose_7 = x_61.transpose(1, 0)
        x_61 = None
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
        x_62 = transpose_8.reshape(16, 49, -1)
        transpose_8 = None
        x_63 = torch._C._nn.linear(
            x_62,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_62 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_64 = torch.nn.functional.dropout(x_63, 0.0, False, False)
        x_63 = None
        attn_windows_2 = x_64.view(-1, 7, 7, 192)
        x_64 = None
        x_65 = attn_windows_2.view(-1, 4, 4, 7, 7, 192)
        attn_windows_2 = None
        permute_14 = x_65.permute(0, 1, 3, 2, 4, 5)
        x_65 = None
        contiguous_7 = permute_14.contiguous()
        permute_14 = None
        x_66 = contiguous_7.view(-1, 28, 28, 192)
        contiguous_7 = None
        getitem_15 = x_66[
            (
                slice(None, None, None),
                slice(None, 28, None),
                slice(None, 28, None),
                slice(None, None, None),
            )
        ]
        x_66 = None
        x_67 = getitem_15.contiguous()
        getitem_15 = None
        layer_norm_6 = torch.nn.functional.layer_norm(
            x_67,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_67 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (None)
        x_68 = x_54 + layer_norm_6
        x_54 = layer_norm_6 = None
        x_69 = x_68.reshape(1, -1, 192)
        x_68 = None
        x_70 = torch._C._nn.linear(
            x_69,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_71 = torch._C._nn.gelu(x_70, approximate="none")
        x_70 = None
        x_72 = torch.nn.functional.dropout(x_71, 0.0, False, False)
        x_71 = None
        x_73 = torch._C._nn.linear(
            x_72,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_72 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_74 = torch.nn.functional.dropout(x_73, 0.0, False, False)
        x_73 = None
        layer_norm_7 = torch.nn.functional.layer_norm(
            x_74,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_74 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_75 = x_69 + layer_norm_7
        x_69 = layer_norm_7 = None
        x_76 = x_75.reshape(1, 28, 28, 192)
        x_75 = None
        x_77 = torch.roll(x_76, shifts=(-3, -3), dims=(1, 2))
        x_78 = torch._C._nn.pad(x_77, (0, 0, 0, 0, 0, 0), "constant", None)
        x_77 = None
        x_79 = x_78.view(1, 4, 7, 4, 7, 192)
        x_78 = None
        permute_15 = x_79.permute(0, 1, 3, 2, 4, 5)
        x_79 = None
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
        x_80 = torch._C._nn.linear(
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_81 = torch.nn.functional.relu(x_80, inplace=False)
        x_80 = None
        x_82 = torch.nn.functional.dropout(x_81, 0.125, False, False)
        x_81 = None
        x_83 = torch._C._nn.linear(
            x_82,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_82 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_84 = torch.nn.functional.dropout(x_83, 0.0, False, False)
        x_83 = None
        transpose_10 = x_84.transpose(1, 0)
        x_84 = None
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
        x_85 = transpose_11.reshape(16, 49, -1)
        transpose_11 = None
        x_86 = torch._C._nn.linear(
            x_85,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_85 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_87 = torch.nn.functional.dropout(x_86, 0.0, False, False)
        x_86 = None
        attn_windows_3 = x_87.view(-1, 7, 7, 192)
        x_87 = None
        x_88 = attn_windows_3.view(-1, 4, 4, 7, 7, 192)
        attn_windows_3 = None
        permute_17 = x_88.permute(0, 1, 3, 2, 4, 5)
        x_88 = None
        contiguous_10 = permute_17.contiguous()
        permute_17 = None
        x_89 = contiguous_10.view(-1, 28, 28, 192)
        contiguous_10 = None
        getitem_19 = x_89[
            (
                slice(None, None, None),
                slice(None, 28, None),
                slice(None, 28, None),
                slice(None, None, None),
            )
        ]
        x_89 = None
        x_90 = getitem_19.contiguous()
        getitem_19 = None
        x_91 = torch.roll(x_90, shifts=(3, 3), dims=(1, 2))
        x_90 = None
        layer_norm_8 = torch.nn.functional.layer_norm(
            x_91,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_91 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_92 = x_76 + layer_norm_8
        x_76 = layer_norm_8 = None
        x_93 = x_92.reshape(1, -1, 192)
        x_92 = None
        x_94 = torch._C._nn.linear(
            x_93,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_95 = torch._C._nn.gelu(x_94, approximate="none")
        x_94 = None
        x_96 = torch.nn.functional.dropout(x_95, 0.0, False, False)
        x_95 = None
        x_97 = torch._C._nn.linear(
            x_96,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_96 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_98 = torch.nn.functional.dropout(x_97, 0.0, False, False)
        x_97 = None
        layer_norm_9 = torch.nn.functional.layer_norm(
            x_98,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_98 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_99 = x_93 + layer_norm_9
        x_93 = layer_norm_9 = None
        x_100 = x_99.reshape(1, 28, 28, 192)
        x_99 = None
        x_101 = x_100.permute(0, 3, 1, 2)
        x_100 = None
        x_102 = x_101.permute(0, 2, 3, 1)
        x_101 = None
        x_103 = torch._C._nn.pad(x_102, (0, 0, 0, 0, 0, 0), "constant", None)
        x_102 = None
        reshape_21 = x_103.reshape(1, 14, 2, 14, 2, 192)
        x_103 = None
        permute_20 = reshape_21.permute(0, 1, 3, 4, 2, 5)
        reshape_21 = None
        x_104 = permute_20.flatten(3)
        permute_20 = None
        x_105 = torch.nn.functional.layer_norm(
            x_104,
            (768,),
            l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_bias_,
            1e-05,
        )
        x_104 = l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_bias_ = (None)
        x_106 = torch._C._nn.linear(
            x_105,
            l_self_modules_stages_modules_2_modules_downsample_modules_reduction_parameters_weight_,
            None,
        )
        x_105 = l_self_modules_stages_modules_2_modules_downsample_modules_reduction_parameters_weight_ = (None)
        x_107 = torch._C._nn.pad(x_106, (0, 0, 0, 0, 0, 0), "constant", None)
        x_108 = x_107.view(1, 2, 7, 2, 7, 384)
        x_107 = None
        permute_21 = x_108.permute(0, 1, 3, 2, 4, 5)
        x_108 = None
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
        x_109 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_110 = torch.nn.functional.relu(x_109, inplace=False)
        x_109 = None
        x_111 = torch.nn.functional.dropout(x_110, 0.125, False, False)
        x_110 = None
        x_112 = torch._C._nn.linear(
            x_111,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_111 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_113 = torch.nn.functional.dropout(x_112, 0.0, False, False)
        x_112 = None
        transpose_13 = x_113.transpose(1, 0)
        x_113 = None
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
        x_114 = transpose_14.reshape(4, 49, -1)
        transpose_14 = None
        x_115 = torch._C._nn.linear(
            x_114,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_114 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_116 = torch.nn.functional.dropout(x_115, 0.0, False, False)
        x_115 = None
        attn_windows_4 = x_116.view(-1, 7, 7, 384)
        x_116 = None
        x_117 = attn_windows_4.view(-1, 2, 2, 7, 7, 384)
        attn_windows_4 = None
        permute_23 = x_117.permute(0, 1, 3, 2, 4, 5)
        x_117 = None
        contiguous_13 = permute_23.contiguous()
        permute_23 = None
        x_118 = contiguous_13.view(-1, 14, 14, 384)
        contiguous_13 = None
        getitem_23 = x_118[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_118 = None
        x_119 = getitem_23.contiguous()
        getitem_23 = None
        layer_norm_11 = torch.nn.functional.layer_norm(
            x_119,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_119 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (None)
        x_120 = x_106 + layer_norm_11
        x_106 = layer_norm_11 = None
        x_121 = x_120.reshape(1, -1, 384)
        x_120 = None
        x_122 = torch._C._nn.linear(
            x_121,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_123 = torch._C._nn.gelu(x_122, approximate="none")
        x_122 = None
        x_124 = torch.nn.functional.dropout(x_123, 0.0, False, False)
        x_123 = None
        x_125 = torch._C._nn.linear(
            x_124,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_124 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_126 = torch.nn.functional.dropout(x_125, 0.0, False, False)
        x_125 = None
        layer_norm_12 = torch.nn.functional.layer_norm(
            x_126,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_126 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_127 = x_121 + layer_norm_12
        x_121 = layer_norm_12 = None
        x_128 = x_127.reshape(1, 14, 14, 384)
        x_127 = None
        x_129 = torch.roll(x_128, shifts=(-3, -3), dims=(1, 2))
        x_130 = torch._C._nn.pad(x_129, (0, 0, 0, 0, 0, 0), "constant", None)
        x_129 = None
        x_131 = x_130.view(1, 2, 7, 2, 7, 384)
        x_130 = None
        permute_24 = x_131.permute(0, 1, 3, 2, 4, 5)
        x_131 = None
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
        x_132 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_133 = torch.nn.functional.relu(x_132, inplace=False)
        x_132 = None
        x_134 = torch.nn.functional.dropout(x_133, 0.125, False, False)
        x_133 = None
        x_135 = torch._C._nn.linear(
            x_134,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_134 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_136 = torch.nn.functional.dropout(x_135, 0.0, False, False)
        x_135 = None
        transpose_16 = x_136.transpose(1, 0)
        x_136 = None
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
        x_137 = transpose_17.reshape(4, 49, -1)
        transpose_17 = None
        x_138 = torch._C._nn.linear(
            x_137,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_137 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_139 = torch.nn.functional.dropout(x_138, 0.0, False, False)
        x_138 = None
        attn_windows_5 = x_139.view(-1, 7, 7, 384)
        x_139 = None
        x_140 = attn_windows_5.view(-1, 2, 2, 7, 7, 384)
        attn_windows_5 = None
        permute_26 = x_140.permute(0, 1, 3, 2, 4, 5)
        x_140 = None
        contiguous_16 = permute_26.contiguous()
        permute_26 = None
        x_141 = contiguous_16.view(-1, 14, 14, 384)
        contiguous_16 = None
        getitem_27 = x_141[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_141 = None
        x_142 = getitem_27.contiguous()
        getitem_27 = None
        x_143 = torch.roll(x_142, shifts=(3, 3), dims=(1, 2))
        x_142 = None
        layer_norm_13 = torch.nn.functional.layer_norm(
            x_143,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_143 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_144 = x_128 + layer_norm_13
        x_128 = layer_norm_13 = None
        x_145 = x_144.reshape(1, -1, 384)
        x_144 = None
        x_146 = torch._C._nn.linear(
            x_145,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_147 = torch._C._nn.gelu(x_146, approximate="none")
        x_146 = None
        x_148 = torch.nn.functional.dropout(x_147, 0.0, False, False)
        x_147 = None
        x_149 = torch._C._nn.linear(
            x_148,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_148 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_150 = torch.nn.functional.dropout(x_149, 0.0, False, False)
        x_149 = None
        layer_norm_14 = torch.nn.functional.layer_norm(
            x_150,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_150 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_151 = x_145 + layer_norm_14
        x_145 = layer_norm_14 = None
        x_152 = x_151.reshape(1, 14, 14, 384)
        x_151 = None
        x_153 = torch._C._nn.pad(x_152, (0, 0, 0, 0, 0, 0), "constant", None)
        x_154 = x_153.view(1, 2, 7, 2, 7, 384)
        x_153 = None
        permute_27 = x_154.permute(0, 1, 3, 2, 4, 5)
        x_154 = None
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
        x_155 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_156 = torch.nn.functional.relu(x_155, inplace=False)
        x_155 = None
        x_157 = torch.nn.functional.dropout(x_156, 0.125, False, False)
        x_156 = None
        x_158 = torch._C._nn.linear(
            x_157,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_157 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_159 = torch.nn.functional.dropout(x_158, 0.0, False, False)
        x_158 = None
        transpose_19 = x_159.transpose(1, 0)
        x_159 = None
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
        x_160 = transpose_20.reshape(4, 49, -1)
        transpose_20 = None
        x_161 = torch._C._nn.linear(
            x_160,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_,
        )
        x_160 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_ = (None)
        x_162 = torch.nn.functional.dropout(x_161, 0.0, False, False)
        x_161 = None
        attn_windows_6 = x_162.view(-1, 7, 7, 384)
        x_162 = None
        x_163 = attn_windows_6.view(-1, 2, 2, 7, 7, 384)
        attn_windows_6 = None
        permute_29 = x_163.permute(0, 1, 3, 2, 4, 5)
        x_163 = None
        contiguous_19 = permute_29.contiguous()
        permute_29 = None
        x_164 = contiguous_19.view(-1, 14, 14, 384)
        contiguous_19 = None
        getitem_31 = x_164[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_164 = None
        x_165 = getitem_31.contiguous()
        getitem_31 = None
        layer_norm_15 = torch.nn.functional.layer_norm(
            x_165,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_165 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_ = (None)
        x_166 = x_152 + layer_norm_15
        x_152 = layer_norm_15 = None
        x_167 = x_166.reshape(1, -1, 384)
        x_166 = None
        x_168 = torch._C._nn.linear(
            x_167,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_169 = torch._C._nn.gelu(x_168, approximate="none")
        x_168 = None
        x_170 = torch.nn.functional.dropout(x_169, 0.0, False, False)
        x_169 = None
        x_171 = torch._C._nn.linear(
            x_170,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_170 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_172 = torch.nn.functional.dropout(x_171, 0.0, False, False)
        x_171 = None
        layer_norm_16 = torch.nn.functional.layer_norm(
            x_172,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_172 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_ = (None)
        x_173 = x_167 + layer_norm_16
        x_167 = layer_norm_16 = None
        x_174 = x_173.reshape(1, 14, 14, 384)
        x_173 = None
        x_175 = torch.roll(x_174, shifts=(-3, -3), dims=(1, 2))
        x_176 = torch._C._nn.pad(x_175, (0, 0, 0, 0, 0, 0), "constant", None)
        x_175 = None
        x_177 = x_176.view(1, 2, 7, 2, 7, 384)
        x_176 = None
        permute_30 = x_177.permute(0, 1, 3, 2, 4, 5)
        x_177 = None
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
        x_178 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_179 = torch.nn.functional.relu(x_178, inplace=False)
        x_178 = None
        x_180 = torch.nn.functional.dropout(x_179, 0.125, False, False)
        x_179 = None
        x_181 = torch._C._nn.linear(
            x_180,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_180 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_182 = torch.nn.functional.dropout(x_181, 0.0, False, False)
        x_181 = None
        transpose_22 = x_182.transpose(1, 0)
        x_182 = None
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
        x_183 = transpose_23.reshape(4, 49, -1)
        transpose_23 = None
        x_184 = torch._C._nn.linear(
            x_183,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_,
        )
        x_183 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_ = (None)
        x_185 = torch.nn.functional.dropout(x_184, 0.0, False, False)
        x_184 = None
        attn_windows_7 = x_185.view(-1, 7, 7, 384)
        x_185 = None
        x_186 = attn_windows_7.view(-1, 2, 2, 7, 7, 384)
        attn_windows_7 = None
        permute_32 = x_186.permute(0, 1, 3, 2, 4, 5)
        x_186 = None
        contiguous_22 = permute_32.contiguous()
        permute_32 = None
        x_187 = contiguous_22.view(-1, 14, 14, 384)
        contiguous_22 = None
        getitem_35 = x_187[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_187 = None
        x_188 = getitem_35.contiguous()
        getitem_35 = None
        x_189 = torch.roll(x_188, shifts=(3, 3), dims=(1, 2))
        x_188 = None
        layer_norm_17 = torch.nn.functional.layer_norm(
            x_189,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_189 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_ = (None)
        x_190 = x_174 + layer_norm_17
        x_174 = layer_norm_17 = None
        x_191 = x_190.reshape(1, -1, 384)
        x_190 = None
        x_192 = torch._C._nn.linear(
            x_191,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_193 = torch._C._nn.gelu(x_192, approximate="none")
        x_192 = None
        x_194 = torch.nn.functional.dropout(x_193, 0.0, False, False)
        x_193 = None
        x_195 = torch._C._nn.linear(
            x_194,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_194 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_196 = torch.nn.functional.dropout(x_195, 0.0, False, False)
        x_195 = None
        layer_norm_18 = torch.nn.functional.layer_norm(
            x_196,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_196 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_ = (None)
        x_197 = x_191 + layer_norm_18
        x_191 = layer_norm_18 = None
        x_198 = x_197.reshape(1, 14, 14, 384)
        x_197 = None
        x_199 = torch._C._nn.pad(x_198, (0, 0, 0, 0, 0, 0), "constant", None)
        x_200 = x_199.view(1, 2, 7, 2, 7, 384)
        x_199 = None
        permute_33 = x_200.permute(0, 1, 3, 2, 4, 5)
        x_200 = None
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
        x_201 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_202 = torch.nn.functional.relu(x_201, inplace=False)
        x_201 = None
        x_203 = torch.nn.functional.dropout(x_202, 0.125, False, False)
        x_202 = None
        x_204 = torch._C._nn.linear(
            x_203,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_203 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_205 = torch.nn.functional.dropout(x_204, 0.0, False, False)
        x_204 = None
        transpose_25 = x_205.transpose(1, 0)
        x_205 = None
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
        x_206 = transpose_26.reshape(4, 49, -1)
        transpose_26 = None
        x_207 = torch._C._nn.linear(
            x_206,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_,
        )
        x_206 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_ = (None)
        x_208 = torch.nn.functional.dropout(x_207, 0.0, False, False)
        x_207 = None
        attn_windows_8 = x_208.view(-1, 7, 7, 384)
        x_208 = None
        x_209 = attn_windows_8.view(-1, 2, 2, 7, 7, 384)
        attn_windows_8 = None
        permute_35 = x_209.permute(0, 1, 3, 2, 4, 5)
        x_209 = None
        contiguous_25 = permute_35.contiguous()
        permute_35 = None
        x_210 = contiguous_25.view(-1, 14, 14, 384)
        contiguous_25 = None
        getitem_39 = x_210[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_210 = None
        x_211 = getitem_39.contiguous()
        getitem_39 = None
        layer_norm_19 = torch.nn.functional.layer_norm(
            x_211,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_211 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_ = (None)
        x_212 = x_198 + layer_norm_19
        x_198 = layer_norm_19 = None
        x_213 = x_212.reshape(1, -1, 384)
        x_212 = None
        x_214 = torch._C._nn.linear(
            x_213,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_215 = torch._C._nn.gelu(x_214, approximate="none")
        x_214 = None
        x_216 = torch.nn.functional.dropout(x_215, 0.0, False, False)
        x_215 = None
        x_217 = torch._C._nn.linear(
            x_216,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_216 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_218 = torch.nn.functional.dropout(x_217, 0.0, False, False)
        x_217 = None
        layer_norm_20 = torch.nn.functional.layer_norm(
            x_218,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_218 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_ = (None)
        x_219 = x_213 + layer_norm_20
        x_213 = layer_norm_20 = None
        x_220 = x_219.reshape(1, 14, 14, 384)
        x_219 = None
        x_221 = torch.roll(x_220, shifts=(-3, -3), dims=(1, 2))
        x_222 = torch._C._nn.pad(x_221, (0, 0, 0, 0, 0, 0), "constant", None)
        x_221 = None
        x_223 = x_222.view(1, 2, 7, 2, 7, 384)
        x_222 = None
        permute_36 = x_223.permute(0, 1, 3, 2, 4, 5)
        x_223 = None
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
        x_224 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_225 = torch.nn.functional.relu(x_224, inplace=False)
        x_224 = None
        x_226 = torch.nn.functional.dropout(x_225, 0.125, False, False)
        x_225 = None
        x_227 = torch._C._nn.linear(
            x_226,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_226 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_228 = torch.nn.functional.dropout(x_227, 0.0, False, False)
        x_227 = None
        transpose_28 = x_228.transpose(1, 0)
        x_228 = None
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
        x_229 = transpose_29.reshape(4, 49, -1)
        transpose_29 = None
        x_230 = torch._C._nn.linear(
            x_229,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_,
        )
        x_229 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_ = (None)
        x_231 = torch.nn.functional.dropout(x_230, 0.0, False, False)
        x_230 = None
        attn_windows_9 = x_231.view(-1, 7, 7, 384)
        x_231 = None
        x_232 = attn_windows_9.view(-1, 2, 2, 7, 7, 384)
        attn_windows_9 = None
        permute_38 = x_232.permute(0, 1, 3, 2, 4, 5)
        x_232 = None
        contiguous_28 = permute_38.contiguous()
        permute_38 = None
        x_233 = contiguous_28.view(-1, 14, 14, 384)
        contiguous_28 = None
        getitem_43 = x_233[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_233 = None
        x_234 = getitem_43.contiguous()
        getitem_43 = None
        x_235 = torch.roll(x_234, shifts=(3, 3), dims=(1, 2))
        x_234 = None
        layer_norm_21 = torch.nn.functional.layer_norm(
            x_235,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_235 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_ = (None)
        x_236 = x_220 + layer_norm_21
        x_220 = layer_norm_21 = None
        x_237 = x_236.reshape(1, -1, 384)
        x_236 = None
        x_238 = torch._C._nn.linear(
            x_237,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_239 = torch._C._nn.gelu(x_238, approximate="none")
        x_238 = None
        x_240 = torch.nn.functional.dropout(x_239, 0.0, False, False)
        x_239 = None
        x_241 = torch._C._nn.linear(
            x_240,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_240 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_242 = torch.nn.functional.dropout(x_241, 0.0, False, False)
        x_241 = None
        layer_norm_22 = torch.nn.functional.layer_norm(
            x_242,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_242 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_ = (None)
        x_243 = x_237 + layer_norm_22
        x_237 = layer_norm_22 = None
        x_244 = x_243.reshape(1, 14, 14, 384)
        x_243 = None
        x_245 = torch._C._nn.pad(x_244, (0, 0, 0, 0, 0, 0), "constant", None)
        x_246 = x_245.view(1, 2, 7, 2, 7, 384)
        x_245 = None
        permute_39 = x_246.permute(0, 1, 3, 2, 4, 5)
        x_246 = None
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
        x_247 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_248 = torch.nn.functional.relu(x_247, inplace=False)
        x_247 = None
        x_249 = torch.nn.functional.dropout(x_248, 0.125, False, False)
        x_248 = None
        x_250 = torch._C._nn.linear(
            x_249,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_249 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_251 = torch.nn.functional.dropout(x_250, 0.0, False, False)
        x_250 = None
        transpose_31 = x_251.transpose(1, 0)
        x_251 = None
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
        x_252 = transpose_32.reshape(4, 49, -1)
        transpose_32 = None
        x_253 = torch._C._nn.linear(
            x_252,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_,
        )
        x_252 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_ = (None)
        x_254 = torch.nn.functional.dropout(x_253, 0.0, False, False)
        x_253 = None
        attn_windows_10 = x_254.view(-1, 7, 7, 384)
        x_254 = None
        x_255 = attn_windows_10.view(-1, 2, 2, 7, 7, 384)
        attn_windows_10 = None
        permute_41 = x_255.permute(0, 1, 3, 2, 4, 5)
        x_255 = None
        contiguous_31 = permute_41.contiguous()
        permute_41 = None
        x_256 = contiguous_31.view(-1, 14, 14, 384)
        contiguous_31 = None
        getitem_47 = x_256[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_256 = None
        x_257 = getitem_47.contiguous()
        getitem_47 = None
        layer_norm_23 = torch.nn.functional.layer_norm(
            x_257,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_257 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_bias_ = (None)
        x_258 = x_244 + layer_norm_23
        x_244 = layer_norm_23 = None
        x_259 = x_258.reshape(1, -1, 384)
        x_258 = None
        x_260 = torch._C._nn.linear(
            x_259,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_261 = torch._C._nn.gelu(x_260, approximate="none")
        x_260 = None
        x_262 = torch.nn.functional.dropout(x_261, 0.0, False, False)
        x_261 = None
        x_263 = torch._C._nn.linear(
            x_262,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_262 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_264 = torch.nn.functional.dropout(x_263, 0.0, False, False)
        x_263 = None
        layer_norm_24 = torch.nn.functional.layer_norm(
            x_264,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_264 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_bias_ = (None)
        x_265 = x_259 + layer_norm_24
        x_259 = layer_norm_24 = None
        x_266 = x_265.reshape(1, 14, 14, 384)
        x_265 = None
        x_267 = torch.roll(x_266, shifts=(-3, -3), dims=(1, 2))
        x_268 = torch._C._nn.pad(x_267, (0, 0, 0, 0, 0, 0), "constant", None)
        x_267 = None
        x_269 = x_268.view(1, 2, 7, 2, 7, 384)
        x_268 = None
        permute_42 = x_269.permute(0, 1, 3, 2, 4, 5)
        x_269 = None
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
        x_270 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_271 = torch.nn.functional.relu(x_270, inplace=False)
        x_270 = None
        x_272 = torch.nn.functional.dropout(x_271, 0.125, False, False)
        x_271 = None
        x_273 = torch._C._nn.linear(
            x_272,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_272 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_274 = torch.nn.functional.dropout(x_273, 0.0, False, False)
        x_273 = None
        transpose_34 = x_274.transpose(1, 0)
        x_274 = None
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
        x_275 = transpose_35.reshape(4, 49, -1)
        transpose_35 = None
        x_276 = torch._C._nn.linear(
            x_275,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_,
        )
        x_275 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_ = (None)
        x_277 = torch.nn.functional.dropout(x_276, 0.0, False, False)
        x_276 = None
        attn_windows_11 = x_277.view(-1, 7, 7, 384)
        x_277 = None
        x_278 = attn_windows_11.view(-1, 2, 2, 7, 7, 384)
        attn_windows_11 = None
        permute_44 = x_278.permute(0, 1, 3, 2, 4, 5)
        x_278 = None
        contiguous_34 = permute_44.contiguous()
        permute_44 = None
        x_279 = contiguous_34.view(-1, 14, 14, 384)
        contiguous_34 = None
        getitem_51 = x_279[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_279 = None
        x_280 = getitem_51.contiguous()
        getitem_51 = None
        x_281 = torch.roll(x_280, shifts=(3, 3), dims=(1, 2))
        x_280 = None
        layer_norm_25 = torch.nn.functional.layer_norm(
            x_281,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_281 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_bias_ = (None)
        x_282 = x_266 + layer_norm_25
        x_266 = layer_norm_25 = None
        x_283 = x_282.reshape(1, -1, 384)
        x_282 = None
        x_284 = torch._C._nn.linear(
            x_283,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_285 = torch._C._nn.gelu(x_284, approximate="none")
        x_284 = None
        x_286 = torch.nn.functional.dropout(x_285, 0.0, False, False)
        x_285 = None
        x_287 = torch._C._nn.linear(
            x_286,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_286 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_288 = torch.nn.functional.dropout(x_287, 0.0, False, False)
        x_287 = None
        layer_norm_26 = torch.nn.functional.layer_norm(
            x_288,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_288 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_bias_ = (None)
        x_289 = x_283 + layer_norm_26
        x_283 = layer_norm_26 = None
        x_290 = x_289.reshape(1, 14, 14, 384)
        x_289 = None
        x_291 = torch._C._nn.pad(x_290, (0, 0, 0, 0, 0, 0), "constant", None)
        x_292 = x_291.view(1, 2, 7, 2, 7, 384)
        x_291 = None
        permute_45 = x_292.permute(0, 1, 3, 2, 4, 5)
        x_292 = None
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
        x_293 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_294 = torch.nn.functional.relu(x_293, inplace=False)
        x_293 = None
        x_295 = torch.nn.functional.dropout(x_294, 0.125, False, False)
        x_294 = None
        x_296 = torch._C._nn.linear(
            x_295,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_295 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_297 = torch.nn.functional.dropout(x_296, 0.0, False, False)
        x_296 = None
        transpose_37 = x_297.transpose(1, 0)
        x_297 = None
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
        x_298 = transpose_38.reshape(4, 49, -1)
        transpose_38 = None
        x_299 = torch._C._nn.linear(
            x_298,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_,
        )
        x_298 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_ = (None)
        x_300 = torch.nn.functional.dropout(x_299, 0.0, False, False)
        x_299 = None
        attn_windows_12 = x_300.view(-1, 7, 7, 384)
        x_300 = None
        x_301 = attn_windows_12.view(-1, 2, 2, 7, 7, 384)
        attn_windows_12 = None
        permute_47 = x_301.permute(0, 1, 3, 2, 4, 5)
        x_301 = None
        contiguous_37 = permute_47.contiguous()
        permute_47 = None
        x_302 = contiguous_37.view(-1, 14, 14, 384)
        contiguous_37 = None
        getitem_55 = x_302[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_302 = None
        x_303 = getitem_55.contiguous()
        getitem_55 = None
        layer_norm_27 = torch.nn.functional.layer_norm(
            x_303,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_303 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_bias_ = (None)
        x_304 = x_290 + layer_norm_27
        x_290 = layer_norm_27 = None
        x_305 = x_304.reshape(1, -1, 384)
        x_304 = None
        x_306 = torch._C._nn.linear(
            x_305,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_307 = torch._C._nn.gelu(x_306, approximate="none")
        x_306 = None
        x_308 = torch.nn.functional.dropout(x_307, 0.0, False, False)
        x_307 = None
        x_309 = torch._C._nn.linear(
            x_308,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_308 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_310 = torch.nn.functional.dropout(x_309, 0.0, False, False)
        x_309 = None
        layer_norm_28 = torch.nn.functional.layer_norm(
            x_310,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_310 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_bias_ = (None)
        x_311 = x_305 + layer_norm_28
        x_305 = layer_norm_28 = None
        x_312 = x_311.reshape(1, 14, 14, 384)
        x_311 = None
        x_313 = torch.roll(x_312, shifts=(-3, -3), dims=(1, 2))
        x_314 = torch._C._nn.pad(x_313, (0, 0, 0, 0, 0, 0), "constant", None)
        x_313 = None
        x_315 = x_314.view(1, 2, 7, 2, 7, 384)
        x_314 = None
        permute_48 = x_315.permute(0, 1, 3, 2, 4, 5)
        x_315 = None
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
        x_316 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_317 = torch.nn.functional.relu(x_316, inplace=False)
        x_316 = None
        x_318 = torch.nn.functional.dropout(x_317, 0.125, False, False)
        x_317 = None
        x_319 = torch._C._nn.linear(
            x_318,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_318 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_320 = torch.nn.functional.dropout(x_319, 0.0, False, False)
        x_319 = None
        transpose_40 = x_320.transpose(1, 0)
        x_320 = None
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
        x_321 = transpose_41.reshape(4, 49, -1)
        transpose_41 = None
        x_322 = torch._C._nn.linear(
            x_321,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_,
        )
        x_321 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_ = (None)
        x_323 = torch.nn.functional.dropout(x_322, 0.0, False, False)
        x_322 = None
        attn_windows_13 = x_323.view(-1, 7, 7, 384)
        x_323 = None
        x_324 = attn_windows_13.view(-1, 2, 2, 7, 7, 384)
        attn_windows_13 = None
        permute_50 = x_324.permute(0, 1, 3, 2, 4, 5)
        x_324 = None
        contiguous_40 = permute_50.contiguous()
        permute_50 = None
        x_325 = contiguous_40.view(-1, 14, 14, 384)
        contiguous_40 = None
        getitem_59 = x_325[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_325 = None
        x_326 = getitem_59.contiguous()
        getitem_59 = None
        x_327 = torch.roll(x_326, shifts=(3, 3), dims=(1, 2))
        x_326 = None
        layer_norm_29 = torch.nn.functional.layer_norm(
            x_327,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_327 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_bias_ = (None)
        x_328 = x_312 + layer_norm_29
        x_312 = layer_norm_29 = None
        x_329 = x_328.reshape(1, -1, 384)
        x_328 = None
        x_330 = torch._C._nn.linear(
            x_329,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_331 = torch._C._nn.gelu(x_330, approximate="none")
        x_330 = None
        x_332 = torch.nn.functional.dropout(x_331, 0.0, False, False)
        x_331 = None
        x_333 = torch._C._nn.linear(
            x_332,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_332 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_334 = torch.nn.functional.dropout(x_333, 0.0, False, False)
        x_333 = None
        layer_norm_30 = torch.nn.functional.layer_norm(
            x_334,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_334 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_bias_ = (None)
        x_335 = x_329 + layer_norm_30
        x_329 = layer_norm_30 = None
        x_336 = x_335.reshape(1, 14, 14, 384)
        x_335 = None
        x_337 = torch._C._nn.pad(x_336, (0, 0, 0, 0, 0, 0), "constant", None)
        x_338 = x_337.view(1, 2, 7, 2, 7, 384)
        x_337 = None
        permute_51 = x_338.permute(0, 1, 3, 2, 4, 5)
        x_338 = None
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
        x_339 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_340 = torch.nn.functional.relu(x_339, inplace=False)
        x_339 = None
        x_341 = torch.nn.functional.dropout(x_340, 0.125, False, False)
        x_340 = None
        x_342 = torch._C._nn.linear(
            x_341,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_341 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_343 = torch.nn.functional.dropout(x_342, 0.0, False, False)
        x_342 = None
        transpose_43 = x_343.transpose(1, 0)
        x_343 = None
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
        x_344 = transpose_44.reshape(4, 49, -1)
        transpose_44 = None
        x_345 = torch._C._nn.linear(
            x_344,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_,
        )
        x_344 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_ = (None)
        x_346 = torch.nn.functional.dropout(x_345, 0.0, False, False)
        x_345 = None
        attn_windows_14 = x_346.view(-1, 7, 7, 384)
        x_346 = None
        x_347 = attn_windows_14.view(-1, 2, 2, 7, 7, 384)
        attn_windows_14 = None
        permute_53 = x_347.permute(0, 1, 3, 2, 4, 5)
        x_347 = None
        contiguous_43 = permute_53.contiguous()
        permute_53 = None
        x_348 = contiguous_43.view(-1, 14, 14, 384)
        contiguous_43 = None
        getitem_63 = x_348[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_348 = None
        x_349 = getitem_63.contiguous()
        getitem_63 = None
        layer_norm_31 = torch.nn.functional.layer_norm(
            x_349,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_349 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_bias_ = (None)
        x_350 = x_336 + layer_norm_31
        x_336 = layer_norm_31 = None
        x_351 = x_350.reshape(1, -1, 384)
        x_350 = None
        x_352 = torch._C._nn.linear(
            x_351,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_353 = torch._C._nn.gelu(x_352, approximate="none")
        x_352 = None
        x_354 = torch.nn.functional.dropout(x_353, 0.0, False, False)
        x_353 = None
        x_355 = torch._C._nn.linear(
            x_354,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_354 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_356 = torch.nn.functional.dropout(x_355, 0.0, False, False)
        x_355 = None
        layer_norm_32 = torch.nn.functional.layer_norm(
            x_356,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_356 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_bias_ = (None)
        x_357 = x_351 + layer_norm_32
        x_351 = layer_norm_32 = None
        x_358 = x_357.reshape(1, 14, 14, 384)
        x_357 = None
        x_359 = torch.roll(x_358, shifts=(-3, -3), dims=(1, 2))
        x_360 = torch._C._nn.pad(x_359, (0, 0, 0, 0, 0, 0), "constant", None)
        x_359 = None
        x_361 = x_360.view(1, 2, 7, 2, 7, 384)
        x_360 = None
        permute_54 = x_361.permute(0, 1, 3, 2, 4, 5)
        x_361 = None
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
        x_362 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_363 = torch.nn.functional.relu(x_362, inplace=False)
        x_362 = None
        x_364 = torch.nn.functional.dropout(x_363, 0.125, False, False)
        x_363 = None
        x_365 = torch._C._nn.linear(
            x_364,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_364 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_366 = torch.nn.functional.dropout(x_365, 0.0, False, False)
        x_365 = None
        transpose_46 = x_366.transpose(1, 0)
        x_366 = None
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
        x_367 = transpose_47.reshape(4, 49, -1)
        transpose_47 = None
        x_368 = torch._C._nn.linear(
            x_367,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_,
        )
        x_367 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_ = (None)
        x_369 = torch.nn.functional.dropout(x_368, 0.0, False, False)
        x_368 = None
        attn_windows_15 = x_369.view(-1, 7, 7, 384)
        x_369 = None
        x_370 = attn_windows_15.view(-1, 2, 2, 7, 7, 384)
        attn_windows_15 = None
        permute_56 = x_370.permute(0, 1, 3, 2, 4, 5)
        x_370 = None
        contiguous_46 = permute_56.contiguous()
        permute_56 = None
        x_371 = contiguous_46.view(-1, 14, 14, 384)
        contiguous_46 = None
        getitem_67 = x_371[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_371 = None
        x_372 = getitem_67.contiguous()
        getitem_67 = None
        x_373 = torch.roll(x_372, shifts=(3, 3), dims=(1, 2))
        x_372 = None
        layer_norm_33 = torch.nn.functional.layer_norm(
            x_373,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_373 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_bias_ = (None)
        x_374 = x_358 + layer_norm_33
        x_358 = layer_norm_33 = None
        x_375 = x_374.reshape(1, -1, 384)
        x_374 = None
        x_376 = torch._C._nn.linear(
            x_375,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_377 = torch._C._nn.gelu(x_376, approximate="none")
        x_376 = None
        x_378 = torch.nn.functional.dropout(x_377, 0.0, False, False)
        x_377 = None
        x_379 = torch._C._nn.linear(
            x_378,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_378 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_380 = torch.nn.functional.dropout(x_379, 0.0, False, False)
        x_379 = None
        layer_norm_34 = torch.nn.functional.layer_norm(
            x_380,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_380 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_bias_ = (None)
        x_381 = x_375 + layer_norm_34
        x_375 = layer_norm_34 = None
        x_382 = x_381.reshape(1, 14, 14, 384)
        x_381 = None
        x_383 = torch._C._nn.pad(x_382, (0, 0, 0, 0, 0, 0), "constant", None)
        x_384 = x_383.view(1, 2, 7, 2, 7, 384)
        x_383 = None
        permute_57 = x_384.permute(0, 1, 3, 2, 4, 5)
        x_384 = None
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
        x_385 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_386 = torch.nn.functional.relu(x_385, inplace=False)
        x_385 = None
        x_387 = torch.nn.functional.dropout(x_386, 0.125, False, False)
        x_386 = None
        x_388 = torch._C._nn.linear(
            x_387,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_387 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_389 = torch.nn.functional.dropout(x_388, 0.0, False, False)
        x_388 = None
        transpose_49 = x_389.transpose(1, 0)
        x_389 = None
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
        x_390 = transpose_50.reshape(4, 49, -1)
        transpose_50 = None
        x_391 = torch._C._nn.linear(
            x_390,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_,
        )
        x_390 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_ = (None)
        x_392 = torch.nn.functional.dropout(x_391, 0.0, False, False)
        x_391 = None
        attn_windows_16 = x_392.view(-1, 7, 7, 384)
        x_392 = None
        x_393 = attn_windows_16.view(-1, 2, 2, 7, 7, 384)
        attn_windows_16 = None
        permute_59 = x_393.permute(0, 1, 3, 2, 4, 5)
        x_393 = None
        contiguous_49 = permute_59.contiguous()
        permute_59 = None
        x_394 = contiguous_49.view(-1, 14, 14, 384)
        contiguous_49 = None
        getitem_71 = x_394[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_394 = None
        x_395 = getitem_71.contiguous()
        getitem_71 = None
        layer_norm_35 = torch.nn.functional.layer_norm(
            x_395,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_395 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_bias_ = (None)
        x_396 = x_382 + layer_norm_35
        x_382 = layer_norm_35 = None
        x_397 = x_396.reshape(1, -1, 384)
        x_396 = None
        x_398 = torch._C._nn.linear(
            x_397,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_399 = torch._C._nn.gelu(x_398, approximate="none")
        x_398 = None
        x_400 = torch.nn.functional.dropout(x_399, 0.0, False, False)
        x_399 = None
        x_401 = torch._C._nn.linear(
            x_400,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_400 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_402 = torch.nn.functional.dropout(x_401, 0.0, False, False)
        x_401 = None
        layer_norm_36 = torch.nn.functional.layer_norm(
            x_402,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_402 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_bias_ = (None)
        x_403 = x_397 + layer_norm_36
        x_397 = layer_norm_36 = None
        x_404 = x_403.reshape(1, 14, 14, 384)
        x_403 = None
        x_405 = torch.roll(x_404, shifts=(-3, -3), dims=(1, 2))
        x_406 = torch._C._nn.pad(x_405, (0, 0, 0, 0, 0, 0), "constant", None)
        x_405 = None
        x_407 = x_406.view(1, 2, 7, 2, 7, 384)
        x_406 = None
        permute_60 = x_407.permute(0, 1, 3, 2, 4, 5)
        x_407 = None
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
        x_408 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_409 = torch.nn.functional.relu(x_408, inplace=False)
        x_408 = None
        x_410 = torch.nn.functional.dropout(x_409, 0.125, False, False)
        x_409 = None
        x_411 = torch._C._nn.linear(
            x_410,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_410 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_412 = torch.nn.functional.dropout(x_411, 0.0, False, False)
        x_411 = None
        transpose_52 = x_412.transpose(1, 0)
        x_412 = None
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
        x_413 = transpose_53.reshape(4, 49, -1)
        transpose_53 = None
        x_414 = torch._C._nn.linear(
            x_413,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_,
        )
        x_413 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_ = (None)
        x_415 = torch.nn.functional.dropout(x_414, 0.0, False, False)
        x_414 = None
        attn_windows_17 = x_415.view(-1, 7, 7, 384)
        x_415 = None
        x_416 = attn_windows_17.view(-1, 2, 2, 7, 7, 384)
        attn_windows_17 = None
        permute_62 = x_416.permute(0, 1, 3, 2, 4, 5)
        x_416 = None
        contiguous_52 = permute_62.contiguous()
        permute_62 = None
        x_417 = contiguous_52.view(-1, 14, 14, 384)
        contiguous_52 = None
        getitem_75 = x_417[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_417 = None
        x_418 = getitem_75.contiguous()
        getitem_75 = None
        x_419 = torch.roll(x_418, shifts=(3, 3), dims=(1, 2))
        x_418 = None
        layer_norm_37 = torch.nn.functional.layer_norm(
            x_419,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_419 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_bias_ = (None)
        x_420 = x_404 + layer_norm_37
        x_404 = layer_norm_37 = None
        x_421 = x_420.reshape(1, -1, 384)
        x_420 = None
        x_422 = torch._C._nn.linear(
            x_421,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_423 = torch._C._nn.gelu(x_422, approximate="none")
        x_422 = None
        x_424 = torch.nn.functional.dropout(x_423, 0.0, False, False)
        x_423 = None
        x_425 = torch._C._nn.linear(
            x_424,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_424 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_426 = torch.nn.functional.dropout(x_425, 0.0, False, False)
        x_425 = None
        layer_norm_38 = torch.nn.functional.layer_norm(
            x_426,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_426 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_bias_ = (None)
        x_427 = x_421 + layer_norm_38
        x_421 = layer_norm_38 = None
        x_428 = x_427.reshape(1, 14, 14, 384)
        x_427 = None
        x_429 = torch._C._nn.pad(x_428, (0, 0, 0, 0, 0, 0), "constant", None)
        x_430 = x_429.view(1, 2, 7, 2, 7, 384)
        x_429 = None
        permute_63 = x_430.permute(0, 1, 3, 2, 4, 5)
        x_430 = None
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
        x_431 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_432 = torch.nn.functional.relu(x_431, inplace=False)
        x_431 = None
        x_433 = torch.nn.functional.dropout(x_432, 0.125, False, False)
        x_432 = None
        x_434 = torch._C._nn.linear(
            x_433,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_433 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_435 = torch.nn.functional.dropout(x_434, 0.0, False, False)
        x_434 = None
        transpose_55 = x_435.transpose(1, 0)
        x_435 = None
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
        x_436 = transpose_56.reshape(4, 49, -1)
        transpose_56 = None
        x_437 = torch._C._nn.linear(
            x_436,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_,
        )
        x_436 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_ = (None)
        x_438 = torch.nn.functional.dropout(x_437, 0.0, False, False)
        x_437 = None
        attn_windows_18 = x_438.view(-1, 7, 7, 384)
        x_438 = None
        x_439 = attn_windows_18.view(-1, 2, 2, 7, 7, 384)
        attn_windows_18 = None
        permute_65 = x_439.permute(0, 1, 3, 2, 4, 5)
        x_439 = None
        contiguous_55 = permute_65.contiguous()
        permute_65 = None
        x_440 = contiguous_55.view(-1, 14, 14, 384)
        contiguous_55 = None
        getitem_79 = x_440[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_440 = None
        x_441 = getitem_79.contiguous()
        getitem_79 = None
        layer_norm_39 = torch.nn.functional.layer_norm(
            x_441,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_441 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_bias_ = (None)
        x_442 = x_428 + layer_norm_39
        x_428 = layer_norm_39 = None
        x_443 = x_442.reshape(1, -1, 384)
        x_442 = None
        x_444 = torch._C._nn.linear(
            x_443,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_445 = torch._C._nn.gelu(x_444, approximate="none")
        x_444 = None
        x_446 = torch.nn.functional.dropout(x_445, 0.0, False, False)
        x_445 = None
        x_447 = torch._C._nn.linear(
            x_446,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_446 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_448 = torch.nn.functional.dropout(x_447, 0.0, False, False)
        x_447 = None
        layer_norm_40 = torch.nn.functional.layer_norm(
            x_448,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_448 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_bias_ = (None)
        x_449 = x_443 + layer_norm_40
        x_443 = layer_norm_40 = None
        x_450 = x_449.reshape(1, 14, 14, 384)
        x_449 = None
        x_451 = torch.roll(x_450, shifts=(-3, -3), dims=(1, 2))
        x_452 = torch._C._nn.pad(x_451, (0, 0, 0, 0, 0, 0), "constant", None)
        x_451 = None
        x_453 = x_452.view(1, 2, 7, 2, 7, 384)
        x_452 = None
        permute_66 = x_453.permute(0, 1, 3, 2, 4, 5)
        x_453 = None
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
        x_454 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_455 = torch.nn.functional.relu(x_454, inplace=False)
        x_454 = None
        x_456 = torch.nn.functional.dropout(x_455, 0.125, False, False)
        x_455 = None
        x_457 = torch._C._nn.linear(
            x_456,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_456 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_458 = torch.nn.functional.dropout(x_457, 0.0, False, False)
        x_457 = None
        transpose_58 = x_458.transpose(1, 0)
        x_458 = None
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
        x_459 = transpose_59.reshape(4, 49, -1)
        transpose_59 = None
        x_460 = torch._C._nn.linear(
            x_459,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_,
        )
        x_459 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_ = (None)
        x_461 = torch.nn.functional.dropout(x_460, 0.0, False, False)
        x_460 = None
        attn_windows_19 = x_461.view(-1, 7, 7, 384)
        x_461 = None
        x_462 = attn_windows_19.view(-1, 2, 2, 7, 7, 384)
        attn_windows_19 = None
        permute_68 = x_462.permute(0, 1, 3, 2, 4, 5)
        x_462 = None
        contiguous_58 = permute_68.contiguous()
        permute_68 = None
        x_463 = contiguous_58.view(-1, 14, 14, 384)
        contiguous_58 = None
        getitem_83 = x_463[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_463 = None
        x_464 = getitem_83.contiguous()
        getitem_83 = None
        x_465 = torch.roll(x_464, shifts=(3, 3), dims=(1, 2))
        x_464 = None
        layer_norm_41 = torch.nn.functional.layer_norm(
            x_465,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_465 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_bias_ = (None)
        x_466 = x_450 + layer_norm_41
        x_450 = layer_norm_41 = None
        x_467 = x_466.reshape(1, -1, 384)
        x_466 = None
        x_468 = torch._C._nn.linear(
            x_467,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_469 = torch._C._nn.gelu(x_468, approximate="none")
        x_468 = None
        x_470 = torch.nn.functional.dropout(x_469, 0.0, False, False)
        x_469 = None
        x_471 = torch._C._nn.linear(
            x_470,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_470 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_472 = torch.nn.functional.dropout(x_471, 0.0, False, False)
        x_471 = None
        layer_norm_42 = torch.nn.functional.layer_norm(
            x_472,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_472 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_bias_ = (None)
        x_473 = x_467 + layer_norm_42
        x_467 = layer_norm_42 = None
        x_474 = x_473.reshape(1, 14, 14, 384)
        x_473 = None
        x_475 = torch._C._nn.pad(x_474, (0, 0, 0, 0, 0, 0), "constant", None)
        x_476 = x_475.view(1, 2, 7, 2, 7, 384)
        x_475 = None
        permute_69 = x_476.permute(0, 1, 3, 2, 4, 5)
        x_476 = None
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
        x_477 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_478 = torch.nn.functional.relu(x_477, inplace=False)
        x_477 = None
        x_479 = torch.nn.functional.dropout(x_478, 0.125, False, False)
        x_478 = None
        x_480 = torch._C._nn.linear(
            x_479,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_479 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_481 = torch.nn.functional.dropout(x_480, 0.0, False, False)
        x_480 = None
        transpose_61 = x_481.transpose(1, 0)
        x_481 = None
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
        x_482 = transpose_62.reshape(4, 49, -1)
        transpose_62 = None
        x_483 = torch._C._nn.linear(
            x_482,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_,
        )
        x_482 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_ = (None)
        x_484 = torch.nn.functional.dropout(x_483, 0.0, False, False)
        x_483 = None
        attn_windows_20 = x_484.view(-1, 7, 7, 384)
        x_484 = None
        x_485 = attn_windows_20.view(-1, 2, 2, 7, 7, 384)
        attn_windows_20 = None
        permute_71 = x_485.permute(0, 1, 3, 2, 4, 5)
        x_485 = None
        contiguous_61 = permute_71.contiguous()
        permute_71 = None
        x_486 = contiguous_61.view(-1, 14, 14, 384)
        contiguous_61 = None
        getitem_87 = x_486[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_486 = None
        x_487 = getitem_87.contiguous()
        getitem_87 = None
        layer_norm_43 = torch.nn.functional.layer_norm(
            x_487,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_487 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_bias_ = (None)
        x_488 = x_474 + layer_norm_43
        x_474 = layer_norm_43 = None
        x_489 = x_488.reshape(1, -1, 384)
        x_488 = None
        x_490 = torch._C._nn.linear(
            x_489,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_491 = torch._C._nn.gelu(x_490, approximate="none")
        x_490 = None
        x_492 = torch.nn.functional.dropout(x_491, 0.0, False, False)
        x_491 = None
        x_493 = torch._C._nn.linear(
            x_492,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_492 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_494 = torch.nn.functional.dropout(x_493, 0.0, False, False)
        x_493 = None
        layer_norm_44 = torch.nn.functional.layer_norm(
            x_494,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_494 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_bias_ = (None)
        x_495 = x_489 + layer_norm_44
        x_489 = layer_norm_44 = None
        x_496 = x_495.reshape(1, 14, 14, 384)
        x_495 = None
        x_497 = torch.roll(x_496, shifts=(-3, -3), dims=(1, 2))
        x_498 = torch._C._nn.pad(x_497, (0, 0, 0, 0, 0, 0), "constant", None)
        x_497 = None
        x_499 = x_498.view(1, 2, 7, 2, 7, 384)
        x_498 = None
        permute_72 = x_499.permute(0, 1, 3, 2, 4, 5)
        x_499 = None
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
        x_500 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_501 = torch.nn.functional.relu(x_500, inplace=False)
        x_500 = None
        x_502 = torch.nn.functional.dropout(x_501, 0.125, False, False)
        x_501 = None
        x_503 = torch._C._nn.linear(
            x_502,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_502 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_504 = torch.nn.functional.dropout(x_503, 0.0, False, False)
        x_503 = None
        transpose_64 = x_504.transpose(1, 0)
        x_504 = None
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
        x_505 = transpose_65.reshape(4, 49, -1)
        transpose_65 = None
        x_506 = torch._C._nn.linear(
            x_505,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_,
        )
        x_505 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_ = (None)
        x_507 = torch.nn.functional.dropout(x_506, 0.0, False, False)
        x_506 = None
        attn_windows_21 = x_507.view(-1, 7, 7, 384)
        x_507 = None
        x_508 = attn_windows_21.view(-1, 2, 2, 7, 7, 384)
        attn_windows_21 = None
        permute_74 = x_508.permute(0, 1, 3, 2, 4, 5)
        x_508 = None
        contiguous_64 = permute_74.contiguous()
        permute_74 = None
        x_509 = contiguous_64.view(-1, 14, 14, 384)
        contiguous_64 = None
        getitem_91 = x_509[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_509 = None
        x_510 = getitem_91.contiguous()
        getitem_91 = None
        x_511 = torch.roll(x_510, shifts=(3, 3), dims=(1, 2))
        x_510 = None
        layer_norm_45 = torch.nn.functional.layer_norm(
            x_511,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_511 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_bias_ = (None)
        x_512 = x_496 + layer_norm_45
        x_496 = layer_norm_45 = None
        x_513 = x_512.reshape(1, -1, 384)
        x_512 = None
        x_514 = torch._C._nn.linear(
            x_513,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_515 = torch._C._nn.gelu(x_514, approximate="none")
        x_514 = None
        x_516 = torch.nn.functional.dropout(x_515, 0.0, False, False)
        x_515 = None
        x_517 = torch._C._nn.linear(
            x_516,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_516 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_518 = torch.nn.functional.dropout(x_517, 0.0, False, False)
        x_517 = None
        layer_norm_46 = torch.nn.functional.layer_norm(
            x_518,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_518 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_bias_ = (None)
        x_519 = x_513 + layer_norm_46
        x_513 = layer_norm_46 = None
        x_520 = x_519.reshape(1, 14, 14, 384)
        x_519 = None
        x_521 = x_520.permute(0, 3, 1, 2)
        x_520 = None
        x_522 = x_521.permute(0, 2, 3, 1)
        x_521 = None
        x_523 = torch._C._nn.pad(x_522, (0, 0, 0, 0, 0, 0), "constant", None)
        x_522 = None
        reshape_112 = x_523.reshape(1, 7, 2, 7, 2, 384)
        x_523 = None
        permute_77 = reshape_112.permute(0, 1, 3, 4, 2, 5)
        reshape_112 = None
        x_524 = permute_77.flatten(3)
        permute_77 = None
        x_525 = torch.nn.functional.layer_norm(
            x_524,
            (1536,),
            l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_bias_,
            1e-05,
        )
        x_524 = l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_ = l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_bias_ = (None)
        x_526 = torch._C._nn.linear(
            x_525,
            l_self_modules_stages_modules_3_modules_downsample_modules_reduction_parameters_weight_,
            None,
        )
        x_525 = l_self_modules_stages_modules_3_modules_downsample_modules_reduction_parameters_weight_ = (None)
        x_527 = torch._C._nn.pad(x_526, (0, 0, 0, 0, 0, 0), "constant", None)
        x_528 = x_527.view(1, 1, 7, 1, 7, 768)
        x_527 = None
        permute_78 = x_528.permute(0, 1, 3, 2, 4, 5)
        x_528 = None
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
        x_529 = torch._C._nn.linear(
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_530 = torch.nn.functional.relu(x_529, inplace=False)
        x_529 = None
        x_531 = torch.nn.functional.dropout(x_530, 0.125, False, False)
        x_530 = None
        x_532 = torch._C._nn.linear(
            x_531,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_531 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_533 = torch.nn.functional.dropout(x_532, 0.0, False, False)
        x_532 = None
        transpose_67 = x_533.transpose(1, 0)
        x_533 = None
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
        x_534 = transpose_68.reshape(1, 49, -1)
        transpose_68 = None
        x_535 = torch._C._nn.linear(
            x_534,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_534 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_536 = torch.nn.functional.dropout(x_535, 0.0, False, False)
        x_535 = None
        attn_windows_22 = x_536.view(-1, 7, 7, 768)
        x_536 = None
        x_537 = attn_windows_22.view(-1, 1, 1, 7, 7, 768)
        attn_windows_22 = None
        permute_80 = x_537.permute(0, 1, 3, 2, 4, 5)
        x_537 = None
        contiguous_67 = permute_80.contiguous()
        permute_80 = None
        x_538 = contiguous_67.view(-1, 7, 7, 768)
        contiguous_67 = None
        getitem_95 = x_538[
            (
                slice(None, None, None),
                slice(None, 7, None),
                slice(None, 7, None),
                slice(None, None, None),
            )
        ]
        x_538 = None
        x_539 = getitem_95.contiguous()
        getitem_95 = None
        layer_norm_48 = torch.nn.functional.layer_norm(
            x_539,
            (768,),
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_539 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (None)
        x_540 = x_526 + layer_norm_48
        x_526 = layer_norm_48 = None
        x_541 = x_540.reshape(1, -1, 768)
        x_540 = None
        x_542 = torch._C._nn.linear(
            x_541,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_543 = torch._C._nn.gelu(x_542, approximate="none")
        x_542 = None
        x_544 = torch.nn.functional.dropout(x_543, 0.0, False, False)
        x_543 = None
        x_545 = torch._C._nn.linear(
            x_544,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_544 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_546 = torch.nn.functional.dropout(x_545, 0.0, False, False)
        x_545 = None
        layer_norm_49 = torch.nn.functional.layer_norm(
            x_546,
            (768,),
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_546 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_547 = x_541 + layer_norm_49
        x_541 = layer_norm_49 = None
        x_548 = x_547.reshape(1, 7, 7, 768)
        x_547 = None
        x_549 = torch._C._nn.pad(x_548, (0, 0, 0, 0, 0, 0), "constant", None)
        x_550 = x_549.view(1, 1, 7, 1, 7, 768)
        x_549 = None
        permute_81 = x_550.permute(0, 1, 3, 2, 4, 5)
        x_550 = None
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
        x_551 = torch._C._nn.linear(
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_buffers_relative_coordinates_log_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_buffers_relative_coordinates_log_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_ = (None)
        x_552 = torch.nn.functional.relu(x_551, inplace=False)
        x_551 = None
        x_553 = torch.nn.functional.dropout(x_552, 0.125, False, False)
        x_552 = None
        x_554 = torch._C._nn.linear(
            x_553,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_,
        )
        x_553 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_ = (None)
        x_555 = torch.nn.functional.dropout(x_554, 0.0, False, False)
        x_554 = None
        transpose_70 = x_555.transpose(1, 0)
        x_555 = None
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
        x_556 = transpose_71.reshape(1, 49, -1)
        transpose_71 = None
        x_557 = torch._C._nn.linear(
            x_556,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_556 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_558 = torch.nn.functional.dropout(x_557, 0.0, False, False)
        x_557 = None
        attn_windows_23 = x_558.view(-1, 7, 7, 768)
        x_558 = None
        x_559 = attn_windows_23.view(-1, 1, 1, 7, 7, 768)
        attn_windows_23 = None
        permute_83 = x_559.permute(0, 1, 3, 2, 4, 5)
        x_559 = None
        contiguous_70 = permute_83.contiguous()
        permute_83 = None
        x_560 = contiguous_70.view(-1, 7, 7, 768)
        contiguous_70 = None
        getitem_99 = x_560[
            (
                slice(None, None, None),
                slice(None, 7, None),
                slice(None, 7, None),
                slice(None, None, None),
            )
        ]
        x_560 = None
        x_561 = getitem_99.contiguous()
        getitem_99 = None
        layer_norm_50 = torch.nn.functional.layer_norm(
            x_561,
            (768,),
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_561 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_562 = x_548 + layer_norm_50
        x_548 = layer_norm_50 = None
        x_563 = x_562.reshape(1, -1, 768)
        x_562 = None
        x_564 = torch._C._nn.linear(
            x_563,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_565 = torch._C._nn.gelu(x_564, approximate="none")
        x_564 = None
        x_566 = torch.nn.functional.dropout(x_565, 0.0, False, False)
        x_565 = None
        x_567 = torch._C._nn.linear(
            x_566,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_566 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_568 = torch.nn.functional.dropout(x_567, 0.0, False, False)
        x_567 = None
        layer_norm_51 = torch.nn.functional.layer_norm(
            x_568,
            (768,),
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_568 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_569 = x_563 + layer_norm_51
        x_563 = layer_norm_51 = None
        x_570 = torch.nn.functional.layer_norm(
            x_569,
            (768,),
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_parameters_bias_,
            1e-05,
        )
        x_569 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_parameters_bias_ = (None)
        x_571 = x_570.reshape(1, 7, 7, 768)
        x_570 = None
        x_572 = x_571.permute(0, 3, 1, 2)
        x_571 = None
        x_573 = torch.nn.functional.adaptive_avg_pool2d(x_572, 1)
        x_572 = None
        x_574 = x_573.flatten(1, -1)
        x_573 = None
        x_575 = torch.nn.functional.dropout(x_574, 0.0, False, False)
        x_574 = None
        x_576 = torch._C._nn.linear(
            x_575,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_575 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_576,)
