import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_x_: torch.Tensor,
        L_self_modules_stem_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_cpe1_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_cpe1_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_cpe2_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_cpe2_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_cpe1_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_cpe1_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_cpe2_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_cpe2_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_cpe1_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_cpe1_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_cpe2_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_cpe2_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_cpe1_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_cpe1_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_cpe2_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_cpe2_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_cpe1_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_cpe1_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_cpe2_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_cpe2_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_cpe1_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_cpe1_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_cpe2_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_cpe2_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_cpe1_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_cpe1_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_cpe2_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_cpe2_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_cpe1_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_cpe1_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_cpe2_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_cpe2_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_cpe1_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_cpe1_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_cpe2_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_cpe2_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_cpe1_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_cpe1_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_cpe2_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_cpe2_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_cpe1_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_cpe1_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_cpe2_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_cpe2_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_cpe1_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_cpe1_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_cpe2_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_cpe2_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_cpe1_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_cpe1_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_cpe2_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_cpe2_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_cpe1_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_cpe1_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_cpe2_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_cpe2_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_cpe1_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_cpe1_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_cpe2_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_cpe2_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_cpe1_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_cpe1_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_cpe2_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_cpe2_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_x_ = L_x_
        l_self_modules_stem_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_conv_parameters_weight_
        )
        l_self_modules_stem_modules_conv_parameters_bias_ = (
            L_self_modules_stem_modules_conv_parameters_bias_
        )
        l_self_modules_stem_modules_norm_parameters_weight_ = (
            L_self_modules_stem_modules_norm_parameters_weight_
        )
        l_self_modules_stem_modules_norm_parameters_bias_ = (
            L_self_modules_stem_modules_norm_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_bias_ = L_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_bias_
        l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_ = L_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_cpe1_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_cpe1_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_cpe1_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_cpe1_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_cpe2_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_cpe2_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_cpe2_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_cpe2_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_cpe1_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_cpe1_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_cpe1_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_cpe1_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_cpe2_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_cpe2_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_cpe2_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_cpe2_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_cpe1_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_cpe1_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_cpe1_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_cpe1_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_cpe2_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_cpe2_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_cpe2_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_cpe2_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_cpe1_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_cpe1_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_cpe1_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_cpe1_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_cpe2_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_cpe2_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_cpe2_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_cpe2_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_cpe1_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_cpe1_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_cpe1_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_cpe1_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_cpe2_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_cpe2_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_cpe2_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_cpe2_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_cpe1_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_cpe1_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_cpe1_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_cpe1_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_cpe2_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_cpe2_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_cpe2_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_cpe2_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_cpe1_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_cpe1_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_cpe1_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_cpe1_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_cpe2_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_cpe2_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_cpe2_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_cpe2_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_cpe1_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_cpe1_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_cpe1_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_cpe1_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_cpe2_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_cpe2_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_cpe2_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_cpe2_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_cpe1_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_cpe1_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_cpe1_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_cpe1_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_cpe2_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_cpe2_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_cpe2_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_cpe2_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_cpe1_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_cpe1_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_cpe1_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_cpe1_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_cpe2_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_cpe2_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_cpe2_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_cpe2_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_cpe1_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_cpe1_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_cpe1_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_cpe1_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_cpe2_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_cpe2_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_cpe2_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_cpe2_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_cpe1_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_cpe1_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_cpe1_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_cpe1_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_cpe2_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_cpe2_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_cpe2_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_cpe2_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_cpe1_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_cpe1_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_cpe1_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_cpe1_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_cpe2_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_cpe2_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_cpe2_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_cpe2_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_cpe1_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_cpe1_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_cpe1_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_cpe1_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_cpe2_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_cpe2_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_cpe2_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_cpe2_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_cpe1_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_cpe1_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_cpe1_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_cpe1_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_cpe2_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_cpe2_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_cpe2_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_cpe2_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_cpe1_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_cpe1_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_cpe1_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_cpe1_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_cpe2_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_cpe2_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_cpe2_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_cpe2_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_ = L_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_
        l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_bias_ = L_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_bias_
        l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_ = L_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_head_modules_norm_parameters_weight_ = (
            L_self_modules_head_modules_norm_parameters_weight_
        )
        l_self_modules_head_modules_norm_parameters_bias_ = (
            L_self_modules_head_modules_norm_parameters_bias_
        )
        l_self_modules_head_modules_fc_parameters_weight_ = (
            L_self_modules_head_modules_fc_parameters_weight_
        )
        l_self_modules_head_modules_fc_parameters_bias_ = (
            L_self_modules_head_modules_fc_parameters_bias_
        )
        x = torch._C._nn.pad(l_x_, (0, 0, 0, 0), "constant", None)
        l_x_ = None
        x_1 = torch.conv2d(
            x,
            l_self_modules_stem_modules_conv_parameters_weight_,
            l_self_modules_stem_modules_conv_parameters_bias_,
            (4, 4),
            (3, 3),
            (1, 1),
            1,
        )
        x = (
            l_self_modules_stem_modules_conv_parameters_weight_
        ) = l_self_modules_stem_modules_conv_parameters_bias_ = None
        x_2 = x_1.permute(0, 2, 3, 1)
        x_1 = None
        x_3 = torch.nn.functional.layer_norm(
            x_2,
            (96,),
            l_self_modules_stem_modules_norm_parameters_weight_,
            l_self_modules_stem_modules_norm_parameters_bias_,
            1e-05,
        )
        x_2 = (
            l_self_modules_stem_modules_norm_parameters_weight_
        ) = l_self_modules_stem_modules_norm_parameters_bias_ = None
        x_4 = x_3.permute(0, 3, 1, 2)
        x_3 = None
        feat = torch.conv2d(
            x_4,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            96,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_bias_ = (None)
        x_5 = x_4 + feat
        x_4 = feat = None
        flatten = x_5.flatten(2)
        x_5 = None
        shortcut = flatten.transpose(1, 2)
        flatten = None
        x_6 = torch.nn.functional.layer_norm(
            shortcut,
            (96,),
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_ = (None)
        x_7 = x_6.view(1, 56, 56, 96)
        x_6 = None
        x_8 = torch._C._nn.pad(x_7, (0, 0, 0, 0, 0, 0), "constant", None)
        x_7 = None
        x_9 = x_8.view(1, 8, 7, 8, 7, 96)
        x_8 = None
        permute_2 = x_9.permute(0, 1, 3, 2, 4, 5)
        x_9 = None
        contiguous = permute_2.contiguous()
        permute_2 = None
        windows = contiguous.view(-1, 7, 7, 96)
        contiguous = None
        x_windows = windows.view(-1, 49, 96)
        windows = None
        linear = torch._C._nn.linear(
            x_windows,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        x_windows = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape = linear.reshape(64, 49, 3, 3, 32)
        linear = None
        qkv = reshape.permute(2, 0, 3, 1, 4)
        reshape = None
        unbind = qkv.unbind(0)
        qkv = None
        q = unbind[0]
        k = unbind[1]
        v = unbind[2]
        unbind = None
        x_10 = torch._C._nn.scaled_dot_product_attention(q, k, v)
        q = k = v = None
        transpose_1 = x_10.transpose(1, 2)
        x_10 = None
        x_11 = transpose_1.reshape(64, 49, 96)
        transpose_1 = None
        x_12 = torch._C._nn.linear(
            x_11,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_11 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        attn_windows = x_12.view(-1, 7, 7, 96)
        x_12 = None
        x_13 = attn_windows.view(-1, 8, 8, 7, 7, 96)
        attn_windows = None
        permute_4 = x_13.permute(0, 1, 3, 2, 4, 5)
        x_13 = None
        contiguous_1 = permute_4.contiguous()
        permute_4 = None
        x_14 = contiguous_1.view(-1, 56, 56, 96)
        contiguous_1 = None
        getitem_3 = x_14[
            (
                slice(None, None, None),
                slice(None, 56, None),
                slice(None, 56, None),
                slice(None, None, None),
            )
        ]
        x_14 = None
        x_15 = getitem_3.contiguous()
        getitem_3 = None
        x_16 = x_15.view(1, 3136, 96)
        x_15 = None
        x_17 = shortcut + x_16
        shortcut = x_16 = None
        transpose_2 = x_17.transpose(1, 2)
        x_17 = None
        view_8 = transpose_2.view(1, 96, 56, 56)
        transpose_2 = None
        feat_1 = torch.conv2d(
            view_8,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            96,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_bias_ = (None)
        x_18 = view_8 + feat_1
        view_8 = feat_1 = None
        flatten_1 = x_18.flatten(2)
        x_18 = None
        x_19 = flatten_1.transpose(1, 2)
        flatten_1 = None
        x_20 = torch.nn.functional.layer_norm(
            x_19,
            (96,),
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_ = (None)
        x_21 = torch._C._nn.linear(
            x_20,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_20 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_22 = torch._C._nn.gelu(x_21, approximate="none")
        x_21 = None
        x_23 = torch.nn.functional.dropout(x_22, 0.0, False, False)
        x_22 = None
        x_24 = torch._C._nn.linear(
            x_23,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_23 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_25 = torch.nn.functional.dropout(x_24, 0.0, False, False)
        x_24 = None
        x_26 = x_19 + x_25
        x_19 = x_25 = None
        transpose_4 = x_26.transpose(1, 2)
        x_26 = None
        x_27 = transpose_4.view(1, 96, 56, 56)
        transpose_4 = None
        feat_2 = torch.conv2d(
            x_27,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            96,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_bias_ = (None)
        x_28 = x_27 + feat_2
        x_27 = feat_2 = None
        flatten_2 = x_28.flatten(2)
        x_28 = None
        x_29 = flatten_2.transpose(1, 2)
        flatten_2 = None
        x_30 = torch.nn.functional.layer_norm(
            x_29,
            (96,),
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_norm1_parameters_bias_ = (None)
        linear_4 = torch._C._nn.linear(
            x_30,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_bias_,
        )
        x_30 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_2 = linear_4.reshape(1, 3136, 3, 3, 32)
        linear_4 = None
        qkv_1 = reshape_2.permute(2, 0, 3, 1, 4)
        reshape_2 = None
        unbind_1 = qkv_1.unbind(0)
        qkv_1 = None
        q_1 = unbind_1[0]
        k_1 = unbind_1[1]
        v_1 = unbind_1[2]
        unbind_1 = None
        k_2 = k_1 * 0.1767766952966369
        k_1 = None
        transpose_6 = k_2.transpose(-1, -2)
        k_2 = None
        attn = transpose_6 @ v_1
        transpose_6 = v_1 = None
        attn_1 = attn.softmax(dim=-1)
        attn = None
        transpose_7 = q_1.transpose(-1, -2)
        q_1 = None
        matmul_1 = attn_1 @ transpose_7
        attn_1 = transpose_7 = None
        x_31 = matmul_1.transpose(-1, -2)
        matmul_1 = None
        transpose_9 = x_31.transpose(1, 2)
        x_31 = None
        x_32 = transpose_9.reshape(1, 3136, 96)
        transpose_9 = None
        x_33 = torch._C._nn.linear(
            x_32,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_32 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_34 = x_29 + x_33
        x_29 = x_33 = None
        transpose_10 = x_34.transpose(1, 2)
        x_34 = None
        view_10 = transpose_10.view(1, 96, 56, 56)
        transpose_10 = None
        feat_3 = torch.conv2d(
            view_10,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            96,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_bias_ = (None)
        x_35 = view_10 + feat_3
        view_10 = feat_3 = None
        flatten_3 = x_35.flatten(2)
        x_35 = None
        x_36 = flatten_3.transpose(1, 2)
        flatten_3 = None
        x_37 = torch.nn.functional.layer_norm(
            x_36,
            (96,),
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_norm2_parameters_bias_ = (None)
        x_38 = torch._C._nn.linear(
            x_37,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_37 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_39 = torch._C._nn.gelu(x_38, approximate="none")
        x_38 = None
        x_40 = torch.nn.functional.dropout(x_39, 0.0, False, False)
        x_39 = None
        x_41 = torch._C._nn.linear(
            x_40,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_40 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_42 = torch.nn.functional.dropout(x_41, 0.0, False, False)
        x_41 = None
        x_43 = x_36 + x_42
        x_36 = x_42 = None
        transpose_12 = x_43.transpose(1, 2)
        x_43 = None
        x_44 = transpose_12.view(1, 96, 56, 56)
        transpose_12 = None
        x_45 = x_44.permute(0, 2, 3, 1)
        x_44 = None
        x_46 = torch.nn.functional.layer_norm(
            x_45,
            (96,),
            l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_bias_,
            1e-05,
        )
        x_45 = l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_bias_ = (None)
        x_47 = x_46.permute(0, 3, 1, 2)
        x_46 = None
        x_48 = torch._C._nn.pad(x_47, (0, 0, 0, 0), "constant", None)
        x_47 = None
        x_49 = torch.conv2d(
            x_48,
            l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_48 = l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_ = l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_ = (None)
        feat_4 = torch.conv2d(
            x_49,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            192,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_bias_ = (None)
        x_50 = x_49 + feat_4
        x_49 = feat_4 = None
        flatten_4 = x_50.flatten(2)
        x_50 = None
        shortcut_1 = flatten_4.transpose(1, 2)
        flatten_4 = None
        x_51 = torch.nn.functional.layer_norm(
            shortcut_1,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_ = (None)
        x_52 = x_51.view(1, 28, 28, 192)
        x_51 = None
        x_53 = torch._C._nn.pad(x_52, (0, 0, 0, 0, 0, 0), "constant", None)
        x_52 = None
        x_54 = x_53.view(1, 4, 7, 4, 7, 192)
        x_53 = None
        permute_8 = x_54.permute(0, 1, 3, 2, 4, 5)
        x_54 = None
        contiguous_3 = permute_8.contiguous()
        permute_8 = None
        windows_1 = contiguous_3.view(-1, 7, 7, 192)
        contiguous_3 = None
        x_windows_1 = windows_1.view(-1, 49, 192)
        windows_1 = None
        linear_8 = torch._C._nn.linear(
            x_windows_1,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        x_windows_1 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_4 = linear_8.reshape(16, 49, 3, 6, 32)
        linear_8 = None
        qkv_2 = reshape_4.permute(2, 0, 3, 1, 4)
        reshape_4 = None
        unbind_2 = qkv_2.unbind(0)
        qkv_2 = None
        q_2 = unbind_2[0]
        k_3 = unbind_2[1]
        v_2 = unbind_2[2]
        unbind_2 = None
        x_55 = torch._C._nn.scaled_dot_product_attention(q_2, k_3, v_2)
        q_2 = k_3 = v_2 = None
        transpose_14 = x_55.transpose(1, 2)
        x_55 = None
        x_56 = transpose_14.reshape(16, 49, 192)
        transpose_14 = None
        x_57 = torch._C._nn.linear(
            x_56,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_56 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        attn_windows_1 = x_57.view(-1, 7, 7, 192)
        x_57 = None
        x_58 = attn_windows_1.view(-1, 4, 4, 7, 7, 192)
        attn_windows_1 = None
        permute_10 = x_58.permute(0, 1, 3, 2, 4, 5)
        x_58 = None
        contiguous_4 = permute_10.contiguous()
        permute_10 = None
        x_59 = contiguous_4.view(-1, 28, 28, 192)
        contiguous_4 = None
        getitem_10 = x_59[
            (
                slice(None, None, None),
                slice(None, 28, None),
                slice(None, 28, None),
                slice(None, None, None),
            )
        ]
        x_59 = None
        x_60 = getitem_10.contiguous()
        getitem_10 = None
        x_61 = x_60.view(1, 784, 192)
        x_60 = None
        x_62 = shortcut_1 + x_61
        shortcut_1 = x_61 = None
        transpose_15 = x_62.transpose(1, 2)
        x_62 = None
        view_20 = transpose_15.view(1, 192, 28, 28)
        transpose_15 = None
        feat_5 = torch.conv2d(
            view_20,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            192,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_bias_ = (None)
        x_63 = view_20 + feat_5
        view_20 = feat_5 = None
        flatten_5 = x_63.flatten(2)
        x_63 = None
        x_64 = flatten_5.transpose(1, 2)
        flatten_5 = None
        x_65 = torch.nn.functional.layer_norm(
            x_64,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_ = (None)
        x_66 = torch._C._nn.linear(
            x_65,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_65 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_67 = torch._C._nn.gelu(x_66, approximate="none")
        x_66 = None
        x_68 = torch.nn.functional.dropout(x_67, 0.0, False, False)
        x_67 = None
        x_69 = torch._C._nn.linear(
            x_68,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_68 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_70 = torch.nn.functional.dropout(x_69, 0.0, False, False)
        x_69 = None
        x_71 = x_64 + x_70
        x_64 = x_70 = None
        transpose_17 = x_71.transpose(1, 2)
        x_71 = None
        x_72 = transpose_17.view(1, 192, 28, 28)
        transpose_17 = None
        feat_6 = torch.conv2d(
            x_72,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            192,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_bias_ = (None)
        x_73 = x_72 + feat_6
        x_72 = feat_6 = None
        flatten_6 = x_73.flatten(2)
        x_73 = None
        x_74 = flatten_6.transpose(1, 2)
        flatten_6 = None
        x_75 = torch.nn.functional.layer_norm(
            x_74,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_norm1_parameters_bias_ = (None)
        linear_12 = torch._C._nn.linear(
            x_75,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_bias_,
        )
        x_75 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_6 = linear_12.reshape(1, 784, 3, 6, 32)
        linear_12 = None
        qkv_3 = reshape_6.permute(2, 0, 3, 1, 4)
        reshape_6 = None
        unbind_3 = qkv_3.unbind(0)
        qkv_3 = None
        q_3 = unbind_3[0]
        k_4 = unbind_3[1]
        v_3 = unbind_3[2]
        unbind_3 = None
        k_5 = k_4 * 0.1767766952966369
        k_4 = None
        transpose_19 = k_5.transpose(-1, -2)
        k_5 = None
        attn_2 = transpose_19 @ v_3
        transpose_19 = v_3 = None
        attn_3 = attn_2.softmax(dim=-1)
        attn_2 = None
        transpose_20 = q_3.transpose(-1, -2)
        q_3 = None
        matmul_3 = attn_3 @ transpose_20
        attn_3 = transpose_20 = None
        x_76 = matmul_3.transpose(-1, -2)
        matmul_3 = None
        transpose_22 = x_76.transpose(1, 2)
        x_76 = None
        x_77 = transpose_22.reshape(1, 784, 192)
        transpose_22 = None
        x_78 = torch._C._nn.linear(
            x_77,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_77 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_79 = x_74 + x_78
        x_74 = x_78 = None
        transpose_23 = x_79.transpose(1, 2)
        x_79 = None
        view_22 = transpose_23.view(1, 192, 28, 28)
        transpose_23 = None
        feat_7 = torch.conv2d(
            view_22,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            192,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_bias_ = (None)
        x_80 = view_22 + feat_7
        view_22 = feat_7 = None
        flatten_7 = x_80.flatten(2)
        x_80 = None
        x_81 = flatten_7.transpose(1, 2)
        flatten_7 = None
        x_82 = torch.nn.functional.layer_norm(
            x_81,
            (192,),
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_norm2_parameters_bias_ = (None)
        x_83 = torch._C._nn.linear(
            x_82,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_82 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_84 = torch._C._nn.gelu(x_83, approximate="none")
        x_83 = None
        x_85 = torch.nn.functional.dropout(x_84, 0.0, False, False)
        x_84 = None
        x_86 = torch._C._nn.linear(
            x_85,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_85 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_87 = torch.nn.functional.dropout(x_86, 0.0, False, False)
        x_86 = None
        x_88 = x_81 + x_87
        x_81 = x_87 = None
        transpose_25 = x_88.transpose(1, 2)
        x_88 = None
        x_89 = transpose_25.view(1, 192, 28, 28)
        transpose_25 = None
        x_90 = x_89.permute(0, 2, 3, 1)
        x_89 = None
        x_91 = torch.nn.functional.layer_norm(
            x_90,
            (192,),
            l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_bias_,
            1e-05,
        )
        x_90 = l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_bias_ = (None)
        x_92 = x_91.permute(0, 3, 1, 2)
        x_91 = None
        x_93 = torch._C._nn.pad(x_92, (0, 0, 0, 0), "constant", None)
        x_92 = None
        x_94 = torch.conv2d(
            x_93,
            l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_93 = l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_ = (None)
        feat_8 = torch.conv2d(
            x_94,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_bias_ = (None)
        x_95 = x_94 + feat_8
        x_94 = feat_8 = None
        flatten_8 = x_95.flatten(2)
        x_95 = None
        shortcut_2 = flatten_8.transpose(1, 2)
        flatten_8 = None
        x_96 = torch.nn.functional.layer_norm(
            shortcut_2,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_ = (None)
        x_97 = x_96.view(1, 14, 14, 384)
        x_96 = None
        x_98 = torch._C._nn.pad(x_97, (0, 0, 0, 0, 0, 0), "constant", None)
        x_97 = None
        x_99 = x_98.view(1, 2, 7, 2, 7, 384)
        x_98 = None
        permute_14 = x_99.permute(0, 1, 3, 2, 4, 5)
        x_99 = None
        contiguous_6 = permute_14.contiguous()
        permute_14 = None
        windows_2 = contiguous_6.view(-1, 7, 7, 384)
        contiguous_6 = None
        x_windows_2 = windows_2.view(-1, 49, 384)
        windows_2 = None
        linear_16 = torch._C._nn.linear(
            x_windows_2,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        x_windows_2 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_8 = linear_16.reshape(4, 49, 3, 12, 32)
        linear_16 = None
        qkv_4 = reshape_8.permute(2, 0, 3, 1, 4)
        reshape_8 = None
        unbind_4 = qkv_4.unbind(0)
        qkv_4 = None
        q_4 = unbind_4[0]
        k_6 = unbind_4[1]
        v_4 = unbind_4[2]
        unbind_4 = None
        x_100 = torch._C._nn.scaled_dot_product_attention(q_4, k_6, v_4)
        q_4 = k_6 = v_4 = None
        transpose_27 = x_100.transpose(1, 2)
        x_100 = None
        x_101 = transpose_27.reshape(4, 49, 384)
        transpose_27 = None
        x_102 = torch._C._nn.linear(
            x_101,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_101 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        attn_windows_2 = x_102.view(-1, 7, 7, 384)
        x_102 = None
        x_103 = attn_windows_2.view(-1, 2, 2, 7, 7, 384)
        attn_windows_2 = None
        permute_16 = x_103.permute(0, 1, 3, 2, 4, 5)
        x_103 = None
        contiguous_7 = permute_16.contiguous()
        permute_16 = None
        x_104 = contiguous_7.view(-1, 14, 14, 384)
        contiguous_7 = None
        getitem_17 = x_104[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_104 = None
        x_105 = getitem_17.contiguous()
        getitem_17 = None
        x_106 = x_105.view(1, 196, 384)
        x_105 = None
        x_107 = shortcut_2 + x_106
        shortcut_2 = x_106 = None
        transpose_28 = x_107.transpose(1, 2)
        x_107 = None
        view_32 = transpose_28.view(1, 384, 14, 14)
        transpose_28 = None
        feat_9 = torch.conv2d(
            view_32,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_bias_ = (None)
        x_108 = view_32 + feat_9
        view_32 = feat_9 = None
        flatten_9 = x_108.flatten(2)
        x_108 = None
        x_109 = flatten_9.transpose(1, 2)
        flatten_9 = None
        x_110 = torch.nn.functional.layer_norm(
            x_109,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_ = (None)
        x_111 = torch._C._nn.linear(
            x_110,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_110 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_112 = torch._C._nn.gelu(x_111, approximate="none")
        x_111 = None
        x_113 = torch.nn.functional.dropout(x_112, 0.0, False, False)
        x_112 = None
        x_114 = torch._C._nn.linear(
            x_113,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_113 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_115 = torch.nn.functional.dropout(x_114, 0.0, False, False)
        x_114 = None
        x_116 = x_109 + x_115
        x_109 = x_115 = None
        transpose_30 = x_116.transpose(1, 2)
        x_116 = None
        x_117 = transpose_30.view(1, 384, 14, 14)
        transpose_30 = None
        feat_10 = torch.conv2d(
            x_117,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_bias_ = (None)
        x_118 = x_117 + feat_10
        x_117 = feat_10 = None
        flatten_10 = x_118.flatten(2)
        x_118 = None
        x_119 = flatten_10.transpose(1, 2)
        flatten_10 = None
        x_120 = torch.nn.functional.layer_norm(
            x_119,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_norm1_parameters_bias_ = (None)
        linear_20 = torch._C._nn.linear(
            x_120,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_bias_,
        )
        x_120 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_10 = linear_20.reshape(1, 196, 3, 12, 32)
        linear_20 = None
        qkv_5 = reshape_10.permute(2, 0, 3, 1, 4)
        reshape_10 = None
        unbind_5 = qkv_5.unbind(0)
        qkv_5 = None
        q_5 = unbind_5[0]
        k_7 = unbind_5[1]
        v_5 = unbind_5[2]
        unbind_5 = None
        k_8 = k_7 * 0.1767766952966369
        k_7 = None
        transpose_32 = k_8.transpose(-1, -2)
        k_8 = None
        attn_4 = transpose_32 @ v_5
        transpose_32 = v_5 = None
        attn_5 = attn_4.softmax(dim=-1)
        attn_4 = None
        transpose_33 = q_5.transpose(-1, -2)
        q_5 = None
        matmul_5 = attn_5 @ transpose_33
        attn_5 = transpose_33 = None
        x_121 = matmul_5.transpose(-1, -2)
        matmul_5 = None
        transpose_35 = x_121.transpose(1, 2)
        x_121 = None
        x_122 = transpose_35.reshape(1, 196, 384)
        transpose_35 = None
        x_123 = torch._C._nn.linear(
            x_122,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_122 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_124 = x_119 + x_123
        x_119 = x_123 = None
        transpose_36 = x_124.transpose(1, 2)
        x_124 = None
        view_34 = transpose_36.view(1, 384, 14, 14)
        transpose_36 = None
        feat_11 = torch.conv2d(
            view_34,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_bias_ = (None)
        x_125 = view_34 + feat_11
        view_34 = feat_11 = None
        flatten_11 = x_125.flatten(2)
        x_125 = None
        x_126 = flatten_11.transpose(1, 2)
        flatten_11 = None
        x_127 = torch.nn.functional.layer_norm(
            x_126,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_norm2_parameters_bias_ = (None)
        x_128 = torch._C._nn.linear(
            x_127,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_127 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_129 = torch._C._nn.gelu(x_128, approximate="none")
        x_128 = None
        x_130 = torch.nn.functional.dropout(x_129, 0.0, False, False)
        x_129 = None
        x_131 = torch._C._nn.linear(
            x_130,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_130 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_132 = torch.nn.functional.dropout(x_131, 0.0, False, False)
        x_131 = None
        x_133 = x_126 + x_132
        x_126 = x_132 = None
        transpose_38 = x_133.transpose(1, 2)
        x_133 = None
        x_134 = transpose_38.view(1, 384, 14, 14)
        transpose_38 = None
        feat_12 = torch.conv2d(
            x_134,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_cpe1_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_cpe1_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_cpe1_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_cpe1_modules_proj_parameters_bias_ = (None)
        x_135 = x_134 + feat_12
        x_134 = feat_12 = None
        flatten_12 = x_135.flatten(2)
        x_135 = None
        shortcut_3 = flatten_12.transpose(1, 2)
        flatten_12 = None
        x_136 = torch.nn.functional.layer_norm(
            shortcut_3,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_norm1_parameters_bias_ = (None)
        x_137 = x_136.view(1, 14, 14, 384)
        x_136 = None
        x_138 = torch._C._nn.pad(x_137, (0, 0, 0, 0, 0, 0), "constant", None)
        x_137 = None
        x_139 = x_138.view(1, 2, 7, 2, 7, 384)
        x_138 = None
        permute_18 = x_139.permute(0, 1, 3, 2, 4, 5)
        x_139 = None
        contiguous_9 = permute_18.contiguous()
        permute_18 = None
        windows_3 = contiguous_9.view(-1, 7, 7, 384)
        contiguous_9 = None
        x_windows_3 = windows_3.view(-1, 49, 384)
        windows_3 = None
        linear_24 = torch._C._nn.linear(
            x_windows_3,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        x_windows_3 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_12 = linear_24.reshape(4, 49, 3, 12, 32)
        linear_24 = None
        qkv_6 = reshape_12.permute(2, 0, 3, 1, 4)
        reshape_12 = None
        unbind_6 = qkv_6.unbind(0)
        qkv_6 = None
        q_6 = unbind_6[0]
        k_9 = unbind_6[1]
        v_6 = unbind_6[2]
        unbind_6 = None
        x_140 = torch._C._nn.scaled_dot_product_attention(q_6, k_9, v_6)
        q_6 = k_9 = v_6 = None
        transpose_40 = x_140.transpose(1, 2)
        x_140 = None
        x_141 = transpose_40.reshape(4, 49, 384)
        transpose_40 = None
        x_142 = torch._C._nn.linear(
            x_141,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_141 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        attn_windows_3 = x_142.view(-1, 7, 7, 384)
        x_142 = None
        x_143 = attn_windows_3.view(-1, 2, 2, 7, 7, 384)
        attn_windows_3 = None
        permute_20 = x_143.permute(0, 1, 3, 2, 4, 5)
        x_143 = None
        contiguous_10 = permute_20.contiguous()
        permute_20 = None
        x_144 = contiguous_10.view(-1, 14, 14, 384)
        contiguous_10 = None
        getitem_24 = x_144[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_144 = None
        x_145 = getitem_24.contiguous()
        getitem_24 = None
        x_146 = x_145.view(1, 196, 384)
        x_145 = None
        x_147 = shortcut_3 + x_146
        shortcut_3 = x_146 = None
        transpose_41 = x_147.transpose(1, 2)
        x_147 = None
        view_44 = transpose_41.view(1, 384, 14, 14)
        transpose_41 = None
        feat_13 = torch.conv2d(
            view_44,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_cpe2_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_cpe2_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_cpe2_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_cpe2_modules_proj_parameters_bias_ = (None)
        x_148 = view_44 + feat_13
        view_44 = feat_13 = None
        flatten_13 = x_148.flatten(2)
        x_148 = None
        x_149 = flatten_13.transpose(1, 2)
        flatten_13 = None
        x_150 = torch.nn.functional.layer_norm(
            x_149,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_norm2_parameters_bias_ = (None)
        x_151 = torch._C._nn.linear(
            x_150,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_150 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_152 = torch._C._nn.gelu(x_151, approximate="none")
        x_151 = None
        x_153 = torch.nn.functional.dropout(x_152, 0.0, False, False)
        x_152 = None
        x_154 = torch._C._nn.linear(
            x_153,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_153 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_155 = torch.nn.functional.dropout(x_154, 0.0, False, False)
        x_154 = None
        x_156 = x_149 + x_155
        x_149 = x_155 = None
        transpose_43 = x_156.transpose(1, 2)
        x_156 = None
        x_157 = transpose_43.view(1, 384, 14, 14)
        transpose_43 = None
        feat_14 = torch.conv2d(
            x_157,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_cpe1_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_cpe1_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_cpe1_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_cpe1_modules_proj_parameters_bias_ = (None)
        x_158 = x_157 + feat_14
        x_157 = feat_14 = None
        flatten_14 = x_158.flatten(2)
        x_158 = None
        x_159 = flatten_14.transpose(1, 2)
        flatten_14 = None
        x_160 = torch.nn.functional.layer_norm(
            x_159,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_norm1_parameters_bias_ = (None)
        linear_28 = torch._C._nn.linear(
            x_160,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_,
        )
        x_160 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_14 = linear_28.reshape(1, 196, 3, 12, 32)
        linear_28 = None
        qkv_7 = reshape_14.permute(2, 0, 3, 1, 4)
        reshape_14 = None
        unbind_7 = qkv_7.unbind(0)
        qkv_7 = None
        q_7 = unbind_7[0]
        k_10 = unbind_7[1]
        v_7 = unbind_7[2]
        unbind_7 = None
        k_11 = k_10 * 0.1767766952966369
        k_10 = None
        transpose_45 = k_11.transpose(-1, -2)
        k_11 = None
        attn_6 = transpose_45 @ v_7
        transpose_45 = v_7 = None
        attn_7 = attn_6.softmax(dim=-1)
        attn_6 = None
        transpose_46 = q_7.transpose(-1, -2)
        q_7 = None
        matmul_7 = attn_7 @ transpose_46
        attn_7 = transpose_46 = None
        x_161 = matmul_7.transpose(-1, -2)
        matmul_7 = None
        transpose_48 = x_161.transpose(1, 2)
        x_161 = None
        x_162 = transpose_48.reshape(1, 196, 384)
        transpose_48 = None
        x_163 = torch._C._nn.linear(
            x_162,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_162 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_164 = x_159 + x_163
        x_159 = x_163 = None
        transpose_49 = x_164.transpose(1, 2)
        x_164 = None
        view_46 = transpose_49.view(1, 384, 14, 14)
        transpose_49 = None
        feat_15 = torch.conv2d(
            view_46,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_cpe2_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_cpe2_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_cpe2_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_cpe2_modules_proj_parameters_bias_ = (None)
        x_165 = view_46 + feat_15
        view_46 = feat_15 = None
        flatten_15 = x_165.flatten(2)
        x_165 = None
        x_166 = flatten_15.transpose(1, 2)
        flatten_15 = None
        x_167 = torch.nn.functional.layer_norm(
            x_166,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_norm2_parameters_bias_ = (None)
        x_168 = torch._C._nn.linear(
            x_167,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_167 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_169 = torch._C._nn.gelu(x_168, approximate="none")
        x_168 = None
        x_170 = torch.nn.functional.dropout(x_169, 0.0, False, False)
        x_169 = None
        x_171 = torch._C._nn.linear(
            x_170,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_170 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_172 = torch.nn.functional.dropout(x_171, 0.0, False, False)
        x_171 = None
        x_173 = x_166 + x_172
        x_166 = x_172 = None
        transpose_51 = x_173.transpose(1, 2)
        x_173 = None
        x_174 = transpose_51.view(1, 384, 14, 14)
        transpose_51 = None
        feat_16 = torch.conv2d(
            x_174,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_cpe1_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_cpe1_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_cpe1_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_cpe1_modules_proj_parameters_bias_ = (None)
        x_175 = x_174 + feat_16
        x_174 = feat_16 = None
        flatten_16 = x_175.flatten(2)
        x_175 = None
        shortcut_4 = flatten_16.transpose(1, 2)
        flatten_16 = None
        x_176 = torch.nn.functional.layer_norm(
            shortcut_4,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_norm1_parameters_bias_ = (None)
        x_177 = x_176.view(1, 14, 14, 384)
        x_176 = None
        x_178 = torch._C._nn.pad(x_177, (0, 0, 0, 0, 0, 0), "constant", None)
        x_177 = None
        x_179 = x_178.view(1, 2, 7, 2, 7, 384)
        x_178 = None
        permute_22 = x_179.permute(0, 1, 3, 2, 4, 5)
        x_179 = None
        contiguous_12 = permute_22.contiguous()
        permute_22 = None
        windows_4 = contiguous_12.view(-1, 7, 7, 384)
        contiguous_12 = None
        x_windows_4 = windows_4.view(-1, 49, 384)
        windows_4 = None
        linear_32 = torch._C._nn.linear(
            x_windows_4,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        x_windows_4 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_16 = linear_32.reshape(4, 49, 3, 12, 32)
        linear_32 = None
        qkv_8 = reshape_16.permute(2, 0, 3, 1, 4)
        reshape_16 = None
        unbind_8 = qkv_8.unbind(0)
        qkv_8 = None
        q_8 = unbind_8[0]
        k_12 = unbind_8[1]
        v_8 = unbind_8[2]
        unbind_8 = None
        x_180 = torch._C._nn.scaled_dot_product_attention(q_8, k_12, v_8)
        q_8 = k_12 = v_8 = None
        transpose_53 = x_180.transpose(1, 2)
        x_180 = None
        x_181 = transpose_53.reshape(4, 49, 384)
        transpose_53 = None
        x_182 = torch._C._nn.linear(
            x_181,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_181 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        attn_windows_4 = x_182.view(-1, 7, 7, 384)
        x_182 = None
        x_183 = attn_windows_4.view(-1, 2, 2, 7, 7, 384)
        attn_windows_4 = None
        permute_24 = x_183.permute(0, 1, 3, 2, 4, 5)
        x_183 = None
        contiguous_13 = permute_24.contiguous()
        permute_24 = None
        x_184 = contiguous_13.view(-1, 14, 14, 384)
        contiguous_13 = None
        getitem_31 = x_184[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_184 = None
        x_185 = getitem_31.contiguous()
        getitem_31 = None
        x_186 = x_185.view(1, 196, 384)
        x_185 = None
        x_187 = shortcut_4 + x_186
        shortcut_4 = x_186 = None
        transpose_54 = x_187.transpose(1, 2)
        x_187 = None
        view_56 = transpose_54.view(1, 384, 14, 14)
        transpose_54 = None
        feat_17 = torch.conv2d(
            view_56,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_cpe2_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_cpe2_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_cpe2_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_cpe2_modules_proj_parameters_bias_ = (None)
        x_188 = view_56 + feat_17
        view_56 = feat_17 = None
        flatten_17 = x_188.flatten(2)
        x_188 = None
        x_189 = flatten_17.transpose(1, 2)
        flatten_17 = None
        x_190 = torch.nn.functional.layer_norm(
            x_189,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_norm2_parameters_bias_ = (None)
        x_191 = torch._C._nn.linear(
            x_190,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_190 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_192 = torch._C._nn.gelu(x_191, approximate="none")
        x_191 = None
        x_193 = torch.nn.functional.dropout(x_192, 0.0, False, False)
        x_192 = None
        x_194 = torch._C._nn.linear(
            x_193,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_193 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_195 = torch.nn.functional.dropout(x_194, 0.0, False, False)
        x_194 = None
        x_196 = x_189 + x_195
        x_189 = x_195 = None
        transpose_56 = x_196.transpose(1, 2)
        x_196 = None
        x_197 = transpose_56.view(1, 384, 14, 14)
        transpose_56 = None
        feat_18 = torch.conv2d(
            x_197,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_cpe1_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_cpe1_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_cpe1_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_cpe1_modules_proj_parameters_bias_ = (None)
        x_198 = x_197 + feat_18
        x_197 = feat_18 = None
        flatten_18 = x_198.flatten(2)
        x_198 = None
        x_199 = flatten_18.transpose(1, 2)
        flatten_18 = None
        x_200 = torch.nn.functional.layer_norm(
            x_199,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_norm1_parameters_bias_ = (None)
        linear_36 = torch._C._nn.linear(
            x_200,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_attn_modules_qkv_parameters_bias_,
        )
        x_200 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_18 = linear_36.reshape(1, 196, 3, 12, 32)
        linear_36 = None
        qkv_9 = reshape_18.permute(2, 0, 3, 1, 4)
        reshape_18 = None
        unbind_9 = qkv_9.unbind(0)
        qkv_9 = None
        q_9 = unbind_9[0]
        k_13 = unbind_9[1]
        v_9 = unbind_9[2]
        unbind_9 = None
        k_14 = k_13 * 0.1767766952966369
        k_13 = None
        transpose_58 = k_14.transpose(-1, -2)
        k_14 = None
        attn_8 = transpose_58 @ v_9
        transpose_58 = v_9 = None
        attn_9 = attn_8.softmax(dim=-1)
        attn_8 = None
        transpose_59 = q_9.transpose(-1, -2)
        q_9 = None
        matmul_9 = attn_9 @ transpose_59
        attn_9 = transpose_59 = None
        x_201 = matmul_9.transpose(-1, -2)
        matmul_9 = None
        transpose_61 = x_201.transpose(1, 2)
        x_201 = None
        x_202 = transpose_61.reshape(1, 196, 384)
        transpose_61 = None
        x_203 = torch._C._nn.linear(
            x_202,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_202 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_204 = x_199 + x_203
        x_199 = x_203 = None
        transpose_62 = x_204.transpose(1, 2)
        x_204 = None
        view_58 = transpose_62.view(1, 384, 14, 14)
        transpose_62 = None
        feat_19 = torch.conv2d(
            view_58,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_cpe2_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_cpe2_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_cpe2_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_cpe2_modules_proj_parameters_bias_ = (None)
        x_205 = view_58 + feat_19
        view_58 = feat_19 = None
        flatten_19 = x_205.flatten(2)
        x_205 = None
        x_206 = flatten_19.transpose(1, 2)
        flatten_19 = None
        x_207 = torch.nn.functional.layer_norm(
            x_206,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_norm2_parameters_bias_ = (None)
        x_208 = torch._C._nn.linear(
            x_207,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_207 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_209 = torch._C._nn.gelu(x_208, approximate="none")
        x_208 = None
        x_210 = torch.nn.functional.dropout(x_209, 0.0, False, False)
        x_209 = None
        x_211 = torch._C._nn.linear(
            x_210,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_210 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_212 = torch.nn.functional.dropout(x_211, 0.0, False, False)
        x_211 = None
        x_213 = x_206 + x_212
        x_206 = x_212 = None
        transpose_64 = x_213.transpose(1, 2)
        x_213 = None
        x_214 = transpose_64.view(1, 384, 14, 14)
        transpose_64 = None
        feat_20 = torch.conv2d(
            x_214,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_cpe1_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_cpe1_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_cpe1_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_cpe1_modules_proj_parameters_bias_ = (None)
        x_215 = x_214 + feat_20
        x_214 = feat_20 = None
        flatten_20 = x_215.flatten(2)
        x_215 = None
        shortcut_5 = flatten_20.transpose(1, 2)
        flatten_20 = None
        x_216 = torch.nn.functional.layer_norm(
            shortcut_5,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_norm1_parameters_bias_ = (None)
        x_217 = x_216.view(1, 14, 14, 384)
        x_216 = None
        x_218 = torch._C._nn.pad(x_217, (0, 0, 0, 0, 0, 0), "constant", None)
        x_217 = None
        x_219 = x_218.view(1, 2, 7, 2, 7, 384)
        x_218 = None
        permute_26 = x_219.permute(0, 1, 3, 2, 4, 5)
        x_219 = None
        contiguous_15 = permute_26.contiguous()
        permute_26 = None
        windows_5 = contiguous_15.view(-1, 7, 7, 384)
        contiguous_15 = None
        x_windows_5 = windows_5.view(-1, 49, 384)
        windows_5 = None
        linear_40 = torch._C._nn.linear(
            x_windows_5,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        x_windows_5 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_20 = linear_40.reshape(4, 49, 3, 12, 32)
        linear_40 = None
        qkv_10 = reshape_20.permute(2, 0, 3, 1, 4)
        reshape_20 = None
        unbind_10 = qkv_10.unbind(0)
        qkv_10 = None
        q_10 = unbind_10[0]
        k_15 = unbind_10[1]
        v_10 = unbind_10[2]
        unbind_10 = None
        x_220 = torch._C._nn.scaled_dot_product_attention(q_10, k_15, v_10)
        q_10 = k_15 = v_10 = None
        transpose_66 = x_220.transpose(1, 2)
        x_220 = None
        x_221 = transpose_66.reshape(4, 49, 384)
        transpose_66 = None
        x_222 = torch._C._nn.linear(
            x_221,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_221 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        attn_windows_5 = x_222.view(-1, 7, 7, 384)
        x_222 = None
        x_223 = attn_windows_5.view(-1, 2, 2, 7, 7, 384)
        attn_windows_5 = None
        permute_28 = x_223.permute(0, 1, 3, 2, 4, 5)
        x_223 = None
        contiguous_16 = permute_28.contiguous()
        permute_28 = None
        x_224 = contiguous_16.view(-1, 14, 14, 384)
        contiguous_16 = None
        getitem_38 = x_224[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_224 = None
        x_225 = getitem_38.contiguous()
        getitem_38 = None
        x_226 = x_225.view(1, 196, 384)
        x_225 = None
        x_227 = shortcut_5 + x_226
        shortcut_5 = x_226 = None
        transpose_67 = x_227.transpose(1, 2)
        x_227 = None
        view_68 = transpose_67.view(1, 384, 14, 14)
        transpose_67 = None
        feat_21 = torch.conv2d(
            view_68,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_cpe2_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_cpe2_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_cpe2_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_cpe2_modules_proj_parameters_bias_ = (None)
        x_228 = view_68 + feat_21
        view_68 = feat_21 = None
        flatten_21 = x_228.flatten(2)
        x_228 = None
        x_229 = flatten_21.transpose(1, 2)
        flatten_21 = None
        x_230 = torch.nn.functional.layer_norm(
            x_229,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_norm2_parameters_bias_ = (None)
        x_231 = torch._C._nn.linear(
            x_230,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_230 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_232 = torch._C._nn.gelu(x_231, approximate="none")
        x_231 = None
        x_233 = torch.nn.functional.dropout(x_232, 0.0, False, False)
        x_232 = None
        x_234 = torch._C._nn.linear(
            x_233,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_233 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_235 = torch.nn.functional.dropout(x_234, 0.0, False, False)
        x_234 = None
        x_236 = x_229 + x_235
        x_229 = x_235 = None
        transpose_69 = x_236.transpose(1, 2)
        x_236 = None
        x_237 = transpose_69.view(1, 384, 14, 14)
        transpose_69 = None
        feat_22 = torch.conv2d(
            x_237,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_cpe1_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_cpe1_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_cpe1_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_cpe1_modules_proj_parameters_bias_ = (None)
        x_238 = x_237 + feat_22
        x_237 = feat_22 = None
        flatten_22 = x_238.flatten(2)
        x_238 = None
        x_239 = flatten_22.transpose(1, 2)
        flatten_22 = None
        x_240 = torch.nn.functional.layer_norm(
            x_239,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_norm1_parameters_bias_ = (None)
        linear_44 = torch._C._nn.linear(
            x_240,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_attn_modules_qkv_parameters_bias_,
        )
        x_240 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_22 = linear_44.reshape(1, 196, 3, 12, 32)
        linear_44 = None
        qkv_11 = reshape_22.permute(2, 0, 3, 1, 4)
        reshape_22 = None
        unbind_11 = qkv_11.unbind(0)
        qkv_11 = None
        q_11 = unbind_11[0]
        k_16 = unbind_11[1]
        v_11 = unbind_11[2]
        unbind_11 = None
        k_17 = k_16 * 0.1767766952966369
        k_16 = None
        transpose_71 = k_17.transpose(-1, -2)
        k_17 = None
        attn_10 = transpose_71 @ v_11
        transpose_71 = v_11 = None
        attn_11 = attn_10.softmax(dim=-1)
        attn_10 = None
        transpose_72 = q_11.transpose(-1, -2)
        q_11 = None
        matmul_11 = attn_11 @ transpose_72
        attn_11 = transpose_72 = None
        x_241 = matmul_11.transpose(-1, -2)
        matmul_11 = None
        transpose_74 = x_241.transpose(1, 2)
        x_241 = None
        x_242 = transpose_74.reshape(1, 196, 384)
        transpose_74 = None
        x_243 = torch._C._nn.linear(
            x_242,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_242 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_244 = x_239 + x_243
        x_239 = x_243 = None
        transpose_75 = x_244.transpose(1, 2)
        x_244 = None
        view_70 = transpose_75.view(1, 384, 14, 14)
        transpose_75 = None
        feat_23 = torch.conv2d(
            view_70,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_cpe2_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_cpe2_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_cpe2_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_cpe2_modules_proj_parameters_bias_ = (None)
        x_245 = view_70 + feat_23
        view_70 = feat_23 = None
        flatten_23 = x_245.flatten(2)
        x_245 = None
        x_246 = flatten_23.transpose(1, 2)
        flatten_23 = None
        x_247 = torch.nn.functional.layer_norm(
            x_246,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_norm2_parameters_bias_ = (None)
        x_248 = torch._C._nn.linear(
            x_247,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_247 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_249 = torch._C._nn.gelu(x_248, approximate="none")
        x_248 = None
        x_250 = torch.nn.functional.dropout(x_249, 0.0, False, False)
        x_249 = None
        x_251 = torch._C._nn.linear(
            x_250,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_250 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_252 = torch.nn.functional.dropout(x_251, 0.0, False, False)
        x_251 = None
        x_253 = x_246 + x_252
        x_246 = x_252 = None
        transpose_77 = x_253.transpose(1, 2)
        x_253 = None
        x_254 = transpose_77.view(1, 384, 14, 14)
        transpose_77 = None
        feat_24 = torch.conv2d(
            x_254,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_cpe1_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_cpe1_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_cpe1_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_cpe1_modules_proj_parameters_bias_ = (None)
        x_255 = x_254 + feat_24
        x_254 = feat_24 = None
        flatten_24 = x_255.flatten(2)
        x_255 = None
        shortcut_6 = flatten_24.transpose(1, 2)
        flatten_24 = None
        x_256 = torch.nn.functional.layer_norm(
            shortcut_6,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_norm1_parameters_bias_ = (None)
        x_257 = x_256.view(1, 14, 14, 384)
        x_256 = None
        x_258 = torch._C._nn.pad(x_257, (0, 0, 0, 0, 0, 0), "constant", None)
        x_257 = None
        x_259 = x_258.view(1, 2, 7, 2, 7, 384)
        x_258 = None
        permute_30 = x_259.permute(0, 1, 3, 2, 4, 5)
        x_259 = None
        contiguous_18 = permute_30.contiguous()
        permute_30 = None
        windows_6 = contiguous_18.view(-1, 7, 7, 384)
        contiguous_18 = None
        x_windows_6 = windows_6.view(-1, 49, 384)
        windows_6 = None
        linear_48 = torch._C._nn.linear(
            x_windows_6,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        x_windows_6 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_24 = linear_48.reshape(4, 49, 3, 12, 32)
        linear_48 = None
        qkv_12 = reshape_24.permute(2, 0, 3, 1, 4)
        reshape_24 = None
        unbind_12 = qkv_12.unbind(0)
        qkv_12 = None
        q_12 = unbind_12[0]
        k_18 = unbind_12[1]
        v_12 = unbind_12[2]
        unbind_12 = None
        x_260 = torch._C._nn.scaled_dot_product_attention(q_12, k_18, v_12)
        q_12 = k_18 = v_12 = None
        transpose_79 = x_260.transpose(1, 2)
        x_260 = None
        x_261 = transpose_79.reshape(4, 49, 384)
        transpose_79 = None
        x_262 = torch._C._nn.linear(
            x_261,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_261 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        attn_windows_6 = x_262.view(-1, 7, 7, 384)
        x_262 = None
        x_263 = attn_windows_6.view(-1, 2, 2, 7, 7, 384)
        attn_windows_6 = None
        permute_32 = x_263.permute(0, 1, 3, 2, 4, 5)
        x_263 = None
        contiguous_19 = permute_32.contiguous()
        permute_32 = None
        x_264 = contiguous_19.view(-1, 14, 14, 384)
        contiguous_19 = None
        getitem_45 = x_264[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_264 = None
        x_265 = getitem_45.contiguous()
        getitem_45 = None
        x_266 = x_265.view(1, 196, 384)
        x_265 = None
        x_267 = shortcut_6 + x_266
        shortcut_6 = x_266 = None
        transpose_80 = x_267.transpose(1, 2)
        x_267 = None
        view_80 = transpose_80.view(1, 384, 14, 14)
        transpose_80 = None
        feat_25 = torch.conv2d(
            view_80,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_cpe2_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_cpe2_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_cpe2_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_cpe2_modules_proj_parameters_bias_ = (None)
        x_268 = view_80 + feat_25
        view_80 = feat_25 = None
        flatten_25 = x_268.flatten(2)
        x_268 = None
        x_269 = flatten_25.transpose(1, 2)
        flatten_25 = None
        x_270 = torch.nn.functional.layer_norm(
            x_269,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_norm2_parameters_bias_ = (None)
        x_271 = torch._C._nn.linear(
            x_270,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_270 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_272 = torch._C._nn.gelu(x_271, approximate="none")
        x_271 = None
        x_273 = torch.nn.functional.dropout(x_272, 0.0, False, False)
        x_272 = None
        x_274 = torch._C._nn.linear(
            x_273,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_273 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_275 = torch.nn.functional.dropout(x_274, 0.0, False, False)
        x_274 = None
        x_276 = x_269 + x_275
        x_269 = x_275 = None
        transpose_82 = x_276.transpose(1, 2)
        x_276 = None
        x_277 = transpose_82.view(1, 384, 14, 14)
        transpose_82 = None
        feat_26 = torch.conv2d(
            x_277,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_cpe1_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_cpe1_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_cpe1_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_cpe1_modules_proj_parameters_bias_ = (None)
        x_278 = x_277 + feat_26
        x_277 = feat_26 = None
        flatten_26 = x_278.flatten(2)
        x_278 = None
        x_279 = flatten_26.transpose(1, 2)
        flatten_26 = None
        x_280 = torch.nn.functional.layer_norm(
            x_279,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_norm1_parameters_bias_ = (None)
        linear_52 = torch._C._nn.linear(
            x_280,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_attn_modules_qkv_parameters_bias_,
        )
        x_280 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_26 = linear_52.reshape(1, 196, 3, 12, 32)
        linear_52 = None
        qkv_13 = reshape_26.permute(2, 0, 3, 1, 4)
        reshape_26 = None
        unbind_13 = qkv_13.unbind(0)
        qkv_13 = None
        q_13 = unbind_13[0]
        k_19 = unbind_13[1]
        v_13 = unbind_13[2]
        unbind_13 = None
        k_20 = k_19 * 0.1767766952966369
        k_19 = None
        transpose_84 = k_20.transpose(-1, -2)
        k_20 = None
        attn_12 = transpose_84 @ v_13
        transpose_84 = v_13 = None
        attn_13 = attn_12.softmax(dim=-1)
        attn_12 = None
        transpose_85 = q_13.transpose(-1, -2)
        q_13 = None
        matmul_13 = attn_13 @ transpose_85
        attn_13 = transpose_85 = None
        x_281 = matmul_13.transpose(-1, -2)
        matmul_13 = None
        transpose_87 = x_281.transpose(1, 2)
        x_281 = None
        x_282 = transpose_87.reshape(1, 196, 384)
        transpose_87 = None
        x_283 = torch._C._nn.linear(
            x_282,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_282 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_284 = x_279 + x_283
        x_279 = x_283 = None
        transpose_88 = x_284.transpose(1, 2)
        x_284 = None
        view_82 = transpose_88.view(1, 384, 14, 14)
        transpose_88 = None
        feat_27 = torch.conv2d(
            view_82,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_cpe2_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_cpe2_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_cpe2_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_cpe2_modules_proj_parameters_bias_ = (None)
        x_285 = view_82 + feat_27
        view_82 = feat_27 = None
        flatten_27 = x_285.flatten(2)
        x_285 = None
        x_286 = flatten_27.transpose(1, 2)
        flatten_27 = None
        x_287 = torch.nn.functional.layer_norm(
            x_286,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_norm2_parameters_bias_ = (None)
        x_288 = torch._C._nn.linear(
            x_287,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_287 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_289 = torch._C._nn.gelu(x_288, approximate="none")
        x_288 = None
        x_290 = torch.nn.functional.dropout(x_289, 0.0, False, False)
        x_289 = None
        x_291 = torch._C._nn.linear(
            x_290,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_290 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_292 = torch.nn.functional.dropout(x_291, 0.0, False, False)
        x_291 = None
        x_293 = x_286 + x_292
        x_286 = x_292 = None
        transpose_90 = x_293.transpose(1, 2)
        x_293 = None
        x_294 = transpose_90.view(1, 384, 14, 14)
        transpose_90 = None
        feat_28 = torch.conv2d(
            x_294,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_cpe1_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_cpe1_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_cpe1_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_cpe1_modules_proj_parameters_bias_ = (None)
        x_295 = x_294 + feat_28
        x_294 = feat_28 = None
        flatten_28 = x_295.flatten(2)
        x_295 = None
        shortcut_7 = flatten_28.transpose(1, 2)
        flatten_28 = None
        x_296 = torch.nn.functional.layer_norm(
            shortcut_7,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_norm1_parameters_bias_ = (None)
        x_297 = x_296.view(1, 14, 14, 384)
        x_296 = None
        x_298 = torch._C._nn.pad(x_297, (0, 0, 0, 0, 0, 0), "constant", None)
        x_297 = None
        x_299 = x_298.view(1, 2, 7, 2, 7, 384)
        x_298 = None
        permute_34 = x_299.permute(0, 1, 3, 2, 4, 5)
        x_299 = None
        contiguous_21 = permute_34.contiguous()
        permute_34 = None
        windows_7 = contiguous_21.view(-1, 7, 7, 384)
        contiguous_21 = None
        x_windows_7 = windows_7.view(-1, 49, 384)
        windows_7 = None
        linear_56 = torch._C._nn.linear(
            x_windows_7,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        x_windows_7 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_28 = linear_56.reshape(4, 49, 3, 12, 32)
        linear_56 = None
        qkv_14 = reshape_28.permute(2, 0, 3, 1, 4)
        reshape_28 = None
        unbind_14 = qkv_14.unbind(0)
        qkv_14 = None
        q_14 = unbind_14[0]
        k_21 = unbind_14[1]
        v_14 = unbind_14[2]
        unbind_14 = None
        x_300 = torch._C._nn.scaled_dot_product_attention(q_14, k_21, v_14)
        q_14 = k_21 = v_14 = None
        transpose_92 = x_300.transpose(1, 2)
        x_300 = None
        x_301 = transpose_92.reshape(4, 49, 384)
        transpose_92 = None
        x_302 = torch._C._nn.linear(
            x_301,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_301 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        attn_windows_7 = x_302.view(-1, 7, 7, 384)
        x_302 = None
        x_303 = attn_windows_7.view(-1, 2, 2, 7, 7, 384)
        attn_windows_7 = None
        permute_36 = x_303.permute(0, 1, 3, 2, 4, 5)
        x_303 = None
        contiguous_22 = permute_36.contiguous()
        permute_36 = None
        x_304 = contiguous_22.view(-1, 14, 14, 384)
        contiguous_22 = None
        getitem_52 = x_304[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_304 = None
        x_305 = getitem_52.contiguous()
        getitem_52 = None
        x_306 = x_305.view(1, 196, 384)
        x_305 = None
        x_307 = shortcut_7 + x_306
        shortcut_7 = x_306 = None
        transpose_93 = x_307.transpose(1, 2)
        x_307 = None
        view_92 = transpose_93.view(1, 384, 14, 14)
        transpose_93 = None
        feat_29 = torch.conv2d(
            view_92,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_cpe2_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_cpe2_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_cpe2_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_cpe2_modules_proj_parameters_bias_ = (None)
        x_308 = view_92 + feat_29
        view_92 = feat_29 = None
        flatten_29 = x_308.flatten(2)
        x_308 = None
        x_309 = flatten_29.transpose(1, 2)
        flatten_29 = None
        x_310 = torch.nn.functional.layer_norm(
            x_309,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_norm2_parameters_bias_ = (None)
        x_311 = torch._C._nn.linear(
            x_310,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_310 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_312 = torch._C._nn.gelu(x_311, approximate="none")
        x_311 = None
        x_313 = torch.nn.functional.dropout(x_312, 0.0, False, False)
        x_312 = None
        x_314 = torch._C._nn.linear(
            x_313,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_313 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_315 = torch.nn.functional.dropout(x_314, 0.0, False, False)
        x_314 = None
        x_316 = x_309 + x_315
        x_309 = x_315 = None
        transpose_95 = x_316.transpose(1, 2)
        x_316 = None
        x_317 = transpose_95.view(1, 384, 14, 14)
        transpose_95 = None
        feat_30 = torch.conv2d(
            x_317,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_cpe1_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_cpe1_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_cpe1_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_cpe1_modules_proj_parameters_bias_ = (None)
        x_318 = x_317 + feat_30
        x_317 = feat_30 = None
        flatten_30 = x_318.flatten(2)
        x_318 = None
        x_319 = flatten_30.transpose(1, 2)
        flatten_30 = None
        x_320 = torch.nn.functional.layer_norm(
            x_319,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_norm1_parameters_bias_ = (None)
        linear_60 = torch._C._nn.linear(
            x_320,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_attn_modules_qkv_parameters_bias_,
        )
        x_320 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_30 = linear_60.reshape(1, 196, 3, 12, 32)
        linear_60 = None
        qkv_15 = reshape_30.permute(2, 0, 3, 1, 4)
        reshape_30 = None
        unbind_15 = qkv_15.unbind(0)
        qkv_15 = None
        q_15 = unbind_15[0]
        k_22 = unbind_15[1]
        v_15 = unbind_15[2]
        unbind_15 = None
        k_23 = k_22 * 0.1767766952966369
        k_22 = None
        transpose_97 = k_23.transpose(-1, -2)
        k_23 = None
        attn_14 = transpose_97 @ v_15
        transpose_97 = v_15 = None
        attn_15 = attn_14.softmax(dim=-1)
        attn_14 = None
        transpose_98 = q_15.transpose(-1, -2)
        q_15 = None
        matmul_15 = attn_15 @ transpose_98
        attn_15 = transpose_98 = None
        x_321 = matmul_15.transpose(-1, -2)
        matmul_15 = None
        transpose_100 = x_321.transpose(1, 2)
        x_321 = None
        x_322 = transpose_100.reshape(1, 196, 384)
        transpose_100 = None
        x_323 = torch._C._nn.linear(
            x_322,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_322 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_324 = x_319 + x_323
        x_319 = x_323 = None
        transpose_101 = x_324.transpose(1, 2)
        x_324 = None
        view_94 = transpose_101.view(1, 384, 14, 14)
        transpose_101 = None
        feat_31 = torch.conv2d(
            view_94,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_cpe2_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_cpe2_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_cpe2_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_cpe2_modules_proj_parameters_bias_ = (None)
        x_325 = view_94 + feat_31
        view_94 = feat_31 = None
        flatten_31 = x_325.flatten(2)
        x_325 = None
        x_326 = flatten_31.transpose(1, 2)
        flatten_31 = None
        x_327 = torch.nn.functional.layer_norm(
            x_326,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_norm2_parameters_bias_ = (None)
        x_328 = torch._C._nn.linear(
            x_327,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_327 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_329 = torch._C._nn.gelu(x_328, approximate="none")
        x_328 = None
        x_330 = torch.nn.functional.dropout(x_329, 0.0, False, False)
        x_329 = None
        x_331 = torch._C._nn.linear(
            x_330,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_330 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_332 = torch.nn.functional.dropout(x_331, 0.0, False, False)
        x_331 = None
        x_333 = x_326 + x_332
        x_326 = x_332 = None
        transpose_103 = x_333.transpose(1, 2)
        x_333 = None
        x_334 = transpose_103.view(1, 384, 14, 14)
        transpose_103 = None
        feat_32 = torch.conv2d(
            x_334,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_cpe1_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_cpe1_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_cpe1_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_cpe1_modules_proj_parameters_bias_ = (None)
        x_335 = x_334 + feat_32
        x_334 = feat_32 = None
        flatten_32 = x_335.flatten(2)
        x_335 = None
        shortcut_8 = flatten_32.transpose(1, 2)
        flatten_32 = None
        x_336 = torch.nn.functional.layer_norm(
            shortcut_8,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_norm1_parameters_bias_ = (None)
        x_337 = x_336.view(1, 14, 14, 384)
        x_336 = None
        x_338 = torch._C._nn.pad(x_337, (0, 0, 0, 0, 0, 0), "constant", None)
        x_337 = None
        x_339 = x_338.view(1, 2, 7, 2, 7, 384)
        x_338 = None
        permute_38 = x_339.permute(0, 1, 3, 2, 4, 5)
        x_339 = None
        contiguous_24 = permute_38.contiguous()
        permute_38 = None
        windows_8 = contiguous_24.view(-1, 7, 7, 384)
        contiguous_24 = None
        x_windows_8 = windows_8.view(-1, 49, 384)
        windows_8 = None
        linear_64 = torch._C._nn.linear(
            x_windows_8,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        x_windows_8 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_32 = linear_64.reshape(4, 49, 3, 12, 32)
        linear_64 = None
        qkv_16 = reshape_32.permute(2, 0, 3, 1, 4)
        reshape_32 = None
        unbind_16 = qkv_16.unbind(0)
        qkv_16 = None
        q_16 = unbind_16[0]
        k_24 = unbind_16[1]
        v_16 = unbind_16[2]
        unbind_16 = None
        x_340 = torch._C._nn.scaled_dot_product_attention(q_16, k_24, v_16)
        q_16 = k_24 = v_16 = None
        transpose_105 = x_340.transpose(1, 2)
        x_340 = None
        x_341 = transpose_105.reshape(4, 49, 384)
        transpose_105 = None
        x_342 = torch._C._nn.linear(
            x_341,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_341 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        attn_windows_8 = x_342.view(-1, 7, 7, 384)
        x_342 = None
        x_343 = attn_windows_8.view(-1, 2, 2, 7, 7, 384)
        attn_windows_8 = None
        permute_40 = x_343.permute(0, 1, 3, 2, 4, 5)
        x_343 = None
        contiguous_25 = permute_40.contiguous()
        permute_40 = None
        x_344 = contiguous_25.view(-1, 14, 14, 384)
        contiguous_25 = None
        getitem_59 = x_344[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_344 = None
        x_345 = getitem_59.contiguous()
        getitem_59 = None
        x_346 = x_345.view(1, 196, 384)
        x_345 = None
        x_347 = shortcut_8 + x_346
        shortcut_8 = x_346 = None
        transpose_106 = x_347.transpose(1, 2)
        x_347 = None
        view_104 = transpose_106.view(1, 384, 14, 14)
        transpose_106 = None
        feat_33 = torch.conv2d(
            view_104,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_cpe2_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_cpe2_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_cpe2_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_cpe2_modules_proj_parameters_bias_ = (None)
        x_348 = view_104 + feat_33
        view_104 = feat_33 = None
        flatten_33 = x_348.flatten(2)
        x_348 = None
        x_349 = flatten_33.transpose(1, 2)
        flatten_33 = None
        x_350 = torch.nn.functional.layer_norm(
            x_349,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_norm2_parameters_bias_ = (None)
        x_351 = torch._C._nn.linear(
            x_350,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_350 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_352 = torch._C._nn.gelu(x_351, approximate="none")
        x_351 = None
        x_353 = torch.nn.functional.dropout(x_352, 0.0, False, False)
        x_352 = None
        x_354 = torch._C._nn.linear(
            x_353,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_353 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_355 = torch.nn.functional.dropout(x_354, 0.0, False, False)
        x_354 = None
        x_356 = x_349 + x_355
        x_349 = x_355 = None
        transpose_108 = x_356.transpose(1, 2)
        x_356 = None
        x_357 = transpose_108.view(1, 384, 14, 14)
        transpose_108 = None
        feat_34 = torch.conv2d(
            x_357,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_cpe1_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_cpe1_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_cpe1_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_cpe1_modules_proj_parameters_bias_ = (None)
        x_358 = x_357 + feat_34
        x_357 = feat_34 = None
        flatten_34 = x_358.flatten(2)
        x_358 = None
        x_359 = flatten_34.transpose(1, 2)
        flatten_34 = None
        x_360 = torch.nn.functional.layer_norm(
            x_359,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_norm1_parameters_bias_ = (None)
        linear_68 = torch._C._nn.linear(
            x_360,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_attn_modules_qkv_parameters_bias_,
        )
        x_360 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_34 = linear_68.reshape(1, 196, 3, 12, 32)
        linear_68 = None
        qkv_17 = reshape_34.permute(2, 0, 3, 1, 4)
        reshape_34 = None
        unbind_17 = qkv_17.unbind(0)
        qkv_17 = None
        q_17 = unbind_17[0]
        k_25 = unbind_17[1]
        v_17 = unbind_17[2]
        unbind_17 = None
        k_26 = k_25 * 0.1767766952966369
        k_25 = None
        transpose_110 = k_26.transpose(-1, -2)
        k_26 = None
        attn_16 = transpose_110 @ v_17
        transpose_110 = v_17 = None
        attn_17 = attn_16.softmax(dim=-1)
        attn_16 = None
        transpose_111 = q_17.transpose(-1, -2)
        q_17 = None
        matmul_17 = attn_17 @ transpose_111
        attn_17 = transpose_111 = None
        x_361 = matmul_17.transpose(-1, -2)
        matmul_17 = None
        transpose_113 = x_361.transpose(1, 2)
        x_361 = None
        x_362 = transpose_113.reshape(1, 196, 384)
        transpose_113 = None
        x_363 = torch._C._nn.linear(
            x_362,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_362 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_364 = x_359 + x_363
        x_359 = x_363 = None
        transpose_114 = x_364.transpose(1, 2)
        x_364 = None
        view_106 = transpose_114.view(1, 384, 14, 14)
        transpose_114 = None
        feat_35 = torch.conv2d(
            view_106,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_cpe2_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_cpe2_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_cpe2_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_cpe2_modules_proj_parameters_bias_ = (None)
        x_365 = view_106 + feat_35
        view_106 = feat_35 = None
        flatten_35 = x_365.flatten(2)
        x_365 = None
        x_366 = flatten_35.transpose(1, 2)
        flatten_35 = None
        x_367 = torch.nn.functional.layer_norm(
            x_366,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_norm2_parameters_bias_ = (None)
        x_368 = torch._C._nn.linear(
            x_367,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_367 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_369 = torch._C._nn.gelu(x_368, approximate="none")
        x_368 = None
        x_370 = torch.nn.functional.dropout(x_369, 0.0, False, False)
        x_369 = None
        x_371 = torch._C._nn.linear(
            x_370,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_370 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_372 = torch.nn.functional.dropout(x_371, 0.0, False, False)
        x_371 = None
        x_373 = x_366 + x_372
        x_366 = x_372 = None
        transpose_116 = x_373.transpose(1, 2)
        x_373 = None
        x_374 = transpose_116.view(1, 384, 14, 14)
        transpose_116 = None
        feat_36 = torch.conv2d(
            x_374,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_cpe1_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_cpe1_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_cpe1_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_cpe1_modules_proj_parameters_bias_ = (None)
        x_375 = x_374 + feat_36
        x_374 = feat_36 = None
        flatten_36 = x_375.flatten(2)
        x_375 = None
        shortcut_9 = flatten_36.transpose(1, 2)
        flatten_36 = None
        x_376 = torch.nn.functional.layer_norm(
            shortcut_9,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_norm1_parameters_bias_ = (None)
        x_377 = x_376.view(1, 14, 14, 384)
        x_376 = None
        x_378 = torch._C._nn.pad(x_377, (0, 0, 0, 0, 0, 0), "constant", None)
        x_377 = None
        x_379 = x_378.view(1, 2, 7, 2, 7, 384)
        x_378 = None
        permute_42 = x_379.permute(0, 1, 3, 2, 4, 5)
        x_379 = None
        contiguous_27 = permute_42.contiguous()
        permute_42 = None
        windows_9 = contiguous_27.view(-1, 7, 7, 384)
        contiguous_27 = None
        x_windows_9 = windows_9.view(-1, 49, 384)
        windows_9 = None
        linear_72 = torch._C._nn.linear(
            x_windows_9,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        x_windows_9 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_36 = linear_72.reshape(4, 49, 3, 12, 32)
        linear_72 = None
        qkv_18 = reshape_36.permute(2, 0, 3, 1, 4)
        reshape_36 = None
        unbind_18 = qkv_18.unbind(0)
        qkv_18 = None
        q_18 = unbind_18[0]
        k_27 = unbind_18[1]
        v_18 = unbind_18[2]
        unbind_18 = None
        x_380 = torch._C._nn.scaled_dot_product_attention(q_18, k_27, v_18)
        q_18 = k_27 = v_18 = None
        transpose_118 = x_380.transpose(1, 2)
        x_380 = None
        x_381 = transpose_118.reshape(4, 49, 384)
        transpose_118 = None
        x_382 = torch._C._nn.linear(
            x_381,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_381 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        attn_windows_9 = x_382.view(-1, 7, 7, 384)
        x_382 = None
        x_383 = attn_windows_9.view(-1, 2, 2, 7, 7, 384)
        attn_windows_9 = None
        permute_44 = x_383.permute(0, 1, 3, 2, 4, 5)
        x_383 = None
        contiguous_28 = permute_44.contiguous()
        permute_44 = None
        x_384 = contiguous_28.view(-1, 14, 14, 384)
        contiguous_28 = None
        getitem_66 = x_384[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_384 = None
        x_385 = getitem_66.contiguous()
        getitem_66 = None
        x_386 = x_385.view(1, 196, 384)
        x_385 = None
        x_387 = shortcut_9 + x_386
        shortcut_9 = x_386 = None
        transpose_119 = x_387.transpose(1, 2)
        x_387 = None
        view_116 = transpose_119.view(1, 384, 14, 14)
        transpose_119 = None
        feat_37 = torch.conv2d(
            view_116,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_cpe2_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_cpe2_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_cpe2_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_cpe2_modules_proj_parameters_bias_ = (None)
        x_388 = view_116 + feat_37
        view_116 = feat_37 = None
        flatten_37 = x_388.flatten(2)
        x_388 = None
        x_389 = flatten_37.transpose(1, 2)
        flatten_37 = None
        x_390 = torch.nn.functional.layer_norm(
            x_389,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_norm2_parameters_bias_ = (None)
        x_391 = torch._C._nn.linear(
            x_390,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_390 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_392 = torch._C._nn.gelu(x_391, approximate="none")
        x_391 = None
        x_393 = torch.nn.functional.dropout(x_392, 0.0, False, False)
        x_392 = None
        x_394 = torch._C._nn.linear(
            x_393,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_393 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_395 = torch.nn.functional.dropout(x_394, 0.0, False, False)
        x_394 = None
        x_396 = x_389 + x_395
        x_389 = x_395 = None
        transpose_121 = x_396.transpose(1, 2)
        x_396 = None
        x_397 = transpose_121.view(1, 384, 14, 14)
        transpose_121 = None
        feat_38 = torch.conv2d(
            x_397,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_cpe1_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_cpe1_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_cpe1_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_cpe1_modules_proj_parameters_bias_ = (None)
        x_398 = x_397 + feat_38
        x_397 = feat_38 = None
        flatten_38 = x_398.flatten(2)
        x_398 = None
        x_399 = flatten_38.transpose(1, 2)
        flatten_38 = None
        x_400 = torch.nn.functional.layer_norm(
            x_399,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_norm1_parameters_bias_ = (None)
        linear_76 = torch._C._nn.linear(
            x_400,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_attn_modules_qkv_parameters_bias_,
        )
        x_400 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_38 = linear_76.reshape(1, 196, 3, 12, 32)
        linear_76 = None
        qkv_19 = reshape_38.permute(2, 0, 3, 1, 4)
        reshape_38 = None
        unbind_19 = qkv_19.unbind(0)
        qkv_19 = None
        q_19 = unbind_19[0]
        k_28 = unbind_19[1]
        v_19 = unbind_19[2]
        unbind_19 = None
        k_29 = k_28 * 0.1767766952966369
        k_28 = None
        transpose_123 = k_29.transpose(-1, -2)
        k_29 = None
        attn_18 = transpose_123 @ v_19
        transpose_123 = v_19 = None
        attn_19 = attn_18.softmax(dim=-1)
        attn_18 = None
        transpose_124 = q_19.transpose(-1, -2)
        q_19 = None
        matmul_19 = attn_19 @ transpose_124
        attn_19 = transpose_124 = None
        x_401 = matmul_19.transpose(-1, -2)
        matmul_19 = None
        transpose_126 = x_401.transpose(1, 2)
        x_401 = None
        x_402 = transpose_126.reshape(1, 196, 384)
        transpose_126 = None
        x_403 = torch._C._nn.linear(
            x_402,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_402 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_404 = x_399 + x_403
        x_399 = x_403 = None
        transpose_127 = x_404.transpose(1, 2)
        x_404 = None
        view_118 = transpose_127.view(1, 384, 14, 14)
        transpose_127 = None
        feat_39 = torch.conv2d(
            view_118,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_cpe2_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_cpe2_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_cpe2_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_cpe2_modules_proj_parameters_bias_ = (None)
        x_405 = view_118 + feat_39
        view_118 = feat_39 = None
        flatten_39 = x_405.flatten(2)
        x_405 = None
        x_406 = flatten_39.transpose(1, 2)
        flatten_39 = None
        x_407 = torch.nn.functional.layer_norm(
            x_406,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_norm2_parameters_bias_ = (None)
        x_408 = torch._C._nn.linear(
            x_407,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_407 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_409 = torch._C._nn.gelu(x_408, approximate="none")
        x_408 = None
        x_410 = torch.nn.functional.dropout(x_409, 0.0, False, False)
        x_409 = None
        x_411 = torch._C._nn.linear(
            x_410,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_410 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_412 = torch.nn.functional.dropout(x_411, 0.0, False, False)
        x_411 = None
        x_413 = x_406 + x_412
        x_406 = x_412 = None
        transpose_129 = x_413.transpose(1, 2)
        x_413 = None
        x_414 = transpose_129.view(1, 384, 14, 14)
        transpose_129 = None
        feat_40 = torch.conv2d(
            x_414,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_cpe1_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_cpe1_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_cpe1_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_cpe1_modules_proj_parameters_bias_ = (None)
        x_415 = x_414 + feat_40
        x_414 = feat_40 = None
        flatten_40 = x_415.flatten(2)
        x_415 = None
        shortcut_10 = flatten_40.transpose(1, 2)
        flatten_40 = None
        x_416 = torch.nn.functional.layer_norm(
            shortcut_10,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_norm1_parameters_bias_ = (None)
        x_417 = x_416.view(1, 14, 14, 384)
        x_416 = None
        x_418 = torch._C._nn.pad(x_417, (0, 0, 0, 0, 0, 0), "constant", None)
        x_417 = None
        x_419 = x_418.view(1, 2, 7, 2, 7, 384)
        x_418 = None
        permute_46 = x_419.permute(0, 1, 3, 2, 4, 5)
        x_419 = None
        contiguous_30 = permute_46.contiguous()
        permute_46 = None
        windows_10 = contiguous_30.view(-1, 7, 7, 384)
        contiguous_30 = None
        x_windows_10 = windows_10.view(-1, 49, 384)
        windows_10 = None
        linear_80 = torch._C._nn.linear(
            x_windows_10,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        x_windows_10 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_40 = linear_80.reshape(4, 49, 3, 12, 32)
        linear_80 = None
        qkv_20 = reshape_40.permute(2, 0, 3, 1, 4)
        reshape_40 = None
        unbind_20 = qkv_20.unbind(0)
        qkv_20 = None
        q_20 = unbind_20[0]
        k_30 = unbind_20[1]
        v_20 = unbind_20[2]
        unbind_20 = None
        x_420 = torch._C._nn.scaled_dot_product_attention(q_20, k_30, v_20)
        q_20 = k_30 = v_20 = None
        transpose_131 = x_420.transpose(1, 2)
        x_420 = None
        x_421 = transpose_131.reshape(4, 49, 384)
        transpose_131 = None
        x_422 = torch._C._nn.linear(
            x_421,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_421 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        attn_windows_10 = x_422.view(-1, 7, 7, 384)
        x_422 = None
        x_423 = attn_windows_10.view(-1, 2, 2, 7, 7, 384)
        attn_windows_10 = None
        permute_48 = x_423.permute(0, 1, 3, 2, 4, 5)
        x_423 = None
        contiguous_31 = permute_48.contiguous()
        permute_48 = None
        x_424 = contiguous_31.view(-1, 14, 14, 384)
        contiguous_31 = None
        getitem_73 = x_424[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_424 = None
        x_425 = getitem_73.contiguous()
        getitem_73 = None
        x_426 = x_425.view(1, 196, 384)
        x_425 = None
        x_427 = shortcut_10 + x_426
        shortcut_10 = x_426 = None
        transpose_132 = x_427.transpose(1, 2)
        x_427 = None
        view_128 = transpose_132.view(1, 384, 14, 14)
        transpose_132 = None
        feat_41 = torch.conv2d(
            view_128,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_cpe2_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_cpe2_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_cpe2_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_cpe2_modules_proj_parameters_bias_ = (None)
        x_428 = view_128 + feat_41
        view_128 = feat_41 = None
        flatten_41 = x_428.flatten(2)
        x_428 = None
        x_429 = flatten_41.transpose(1, 2)
        flatten_41 = None
        x_430 = torch.nn.functional.layer_norm(
            x_429,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_norm2_parameters_bias_ = (None)
        x_431 = torch._C._nn.linear(
            x_430,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_430 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_432 = torch._C._nn.gelu(x_431, approximate="none")
        x_431 = None
        x_433 = torch.nn.functional.dropout(x_432, 0.0, False, False)
        x_432 = None
        x_434 = torch._C._nn.linear(
            x_433,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_433 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_435 = torch.nn.functional.dropout(x_434, 0.0, False, False)
        x_434 = None
        x_436 = x_429 + x_435
        x_429 = x_435 = None
        transpose_134 = x_436.transpose(1, 2)
        x_436 = None
        x_437 = transpose_134.view(1, 384, 14, 14)
        transpose_134 = None
        feat_42 = torch.conv2d(
            x_437,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_cpe1_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_cpe1_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_cpe1_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_cpe1_modules_proj_parameters_bias_ = (None)
        x_438 = x_437 + feat_42
        x_437 = feat_42 = None
        flatten_42 = x_438.flatten(2)
        x_438 = None
        x_439 = flatten_42.transpose(1, 2)
        flatten_42 = None
        x_440 = torch.nn.functional.layer_norm(
            x_439,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_norm1_parameters_bias_ = (None)
        linear_84 = torch._C._nn.linear(
            x_440,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_attn_modules_qkv_parameters_bias_,
        )
        x_440 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_42 = linear_84.reshape(1, 196, 3, 12, 32)
        linear_84 = None
        qkv_21 = reshape_42.permute(2, 0, 3, 1, 4)
        reshape_42 = None
        unbind_21 = qkv_21.unbind(0)
        qkv_21 = None
        q_21 = unbind_21[0]
        k_31 = unbind_21[1]
        v_21 = unbind_21[2]
        unbind_21 = None
        k_32 = k_31 * 0.1767766952966369
        k_31 = None
        transpose_136 = k_32.transpose(-1, -2)
        k_32 = None
        attn_20 = transpose_136 @ v_21
        transpose_136 = v_21 = None
        attn_21 = attn_20.softmax(dim=-1)
        attn_20 = None
        transpose_137 = q_21.transpose(-1, -2)
        q_21 = None
        matmul_21 = attn_21 @ transpose_137
        attn_21 = transpose_137 = None
        x_441 = matmul_21.transpose(-1, -2)
        matmul_21 = None
        transpose_139 = x_441.transpose(1, 2)
        x_441 = None
        x_442 = transpose_139.reshape(1, 196, 384)
        transpose_139 = None
        x_443 = torch._C._nn.linear(
            x_442,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_442 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_444 = x_439 + x_443
        x_439 = x_443 = None
        transpose_140 = x_444.transpose(1, 2)
        x_444 = None
        view_130 = transpose_140.view(1, 384, 14, 14)
        transpose_140 = None
        feat_43 = torch.conv2d(
            view_130,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_cpe2_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_cpe2_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_cpe2_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_cpe2_modules_proj_parameters_bias_ = (None)
        x_445 = view_130 + feat_43
        view_130 = feat_43 = None
        flatten_43 = x_445.flatten(2)
        x_445 = None
        x_446 = flatten_43.transpose(1, 2)
        flatten_43 = None
        x_447 = torch.nn.functional.layer_norm(
            x_446,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_norm2_parameters_bias_ = (None)
        x_448 = torch._C._nn.linear(
            x_447,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_447 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_449 = torch._C._nn.gelu(x_448, approximate="none")
        x_448 = None
        x_450 = torch.nn.functional.dropout(x_449, 0.0, False, False)
        x_449 = None
        x_451 = torch._C._nn.linear(
            x_450,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_450 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_452 = torch.nn.functional.dropout(x_451, 0.0, False, False)
        x_451 = None
        x_453 = x_446 + x_452
        x_446 = x_452 = None
        transpose_142 = x_453.transpose(1, 2)
        x_453 = None
        x_454 = transpose_142.view(1, 384, 14, 14)
        transpose_142 = None
        x_455 = x_454.permute(0, 2, 3, 1)
        x_454 = None
        x_456 = torch.nn.functional.layer_norm(
            x_455,
            (384,),
            l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_bias_,
            1e-05,
        )
        x_455 = l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_ = l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_bias_ = (None)
        x_457 = x_456.permute(0, 3, 1, 2)
        x_456 = None
        x_458 = torch._C._nn.pad(x_457, (0, 0, 0, 0), "constant", None)
        x_457 = None
        x_459 = torch.conv2d(
            x_458,
            l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_458 = l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_ = l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_ = (None)
        feat_44 = torch.conv2d(
            x_459,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_bias_ = (None)
        x_460 = x_459 + feat_44
        x_459 = feat_44 = None
        flatten_44 = x_460.flatten(2)
        x_460 = None
        shortcut_11 = flatten_44.transpose(1, 2)
        flatten_44 = None
        x_461 = torch.nn.functional.layer_norm(
            shortcut_11,
            (768,),
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_ = (None)
        x_462 = x_461.view(1, 7, 7, 768)
        x_461 = None
        x_463 = torch._C._nn.pad(x_462, (0, 0, 0, 0, 0, 0), "constant", None)
        x_462 = None
        x_464 = x_463.view(1, 1, 7, 1, 7, 768)
        x_463 = None
        permute_52 = x_464.permute(0, 1, 3, 2, 4, 5)
        x_464 = None
        contiguous_33 = permute_52.contiguous()
        permute_52 = None
        windows_11 = contiguous_33.view(-1, 7, 7, 768)
        contiguous_33 = None
        x_windows_11 = windows_11.view(-1, 49, 768)
        windows_11 = None
        linear_88 = torch._C._nn.linear(
            x_windows_11,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        x_windows_11 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_44 = linear_88.reshape(1, 49, 3, 24, 32)
        linear_88 = None
        qkv_22 = reshape_44.permute(2, 0, 3, 1, 4)
        reshape_44 = None
        unbind_22 = qkv_22.unbind(0)
        qkv_22 = None
        q_22 = unbind_22[0]
        k_33 = unbind_22[1]
        v_22 = unbind_22[2]
        unbind_22 = None
        x_465 = torch._C._nn.scaled_dot_product_attention(q_22, k_33, v_22)
        q_22 = k_33 = v_22 = None
        transpose_144 = x_465.transpose(1, 2)
        x_465 = None
        x_466 = transpose_144.reshape(1, 49, 768)
        transpose_144 = None
        x_467 = torch._C._nn.linear(
            x_466,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_466 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        attn_windows_11 = x_467.view(-1, 7, 7, 768)
        x_467 = None
        x_468 = attn_windows_11.view(-1, 1, 1, 7, 7, 768)
        attn_windows_11 = None
        permute_54 = x_468.permute(0, 1, 3, 2, 4, 5)
        x_468 = None
        contiguous_34 = permute_54.contiguous()
        permute_54 = None
        x_469 = contiguous_34.view(-1, 7, 7, 768)
        contiguous_34 = None
        getitem_80 = x_469[
            (
                slice(None, None, None),
                slice(None, 7, None),
                slice(None, 7, None),
                slice(None, None, None),
            )
        ]
        x_469 = None
        x_470 = getitem_80.contiguous()
        getitem_80 = None
        x_471 = x_470.view(1, 49, 768)
        x_470 = None
        x_472 = shortcut_11 + x_471
        shortcut_11 = x_471 = None
        transpose_145 = x_472.transpose(1, 2)
        x_472 = None
        view_140 = transpose_145.view(1, 768, 7, 7)
        transpose_145 = None
        feat_45 = torch.conv2d(
            view_140,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_bias_ = (None)
        x_473 = view_140 + feat_45
        view_140 = feat_45 = None
        flatten_45 = x_473.flatten(2)
        x_473 = None
        x_474 = flatten_45.transpose(1, 2)
        flatten_45 = None
        x_475 = torch.nn.functional.layer_norm(
            x_474,
            (768,),
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_ = (None)
        x_476 = torch._C._nn.linear(
            x_475,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_475 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_477 = torch._C._nn.gelu(x_476, approximate="none")
        x_476 = None
        x_478 = torch.nn.functional.dropout(x_477, 0.0, False, False)
        x_477 = None
        x_479 = torch._C._nn.linear(
            x_478,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_478 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_480 = torch.nn.functional.dropout(x_479, 0.0, False, False)
        x_479 = None
        x_481 = x_474 + x_480
        x_474 = x_480 = None
        transpose_147 = x_481.transpose(1, 2)
        x_481 = None
        x_482 = transpose_147.view(1, 768, 7, 7)
        transpose_147 = None
        feat_46 = torch.conv2d(
            x_482,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_bias_ = (None)
        x_483 = x_482 + feat_46
        x_482 = feat_46 = None
        flatten_46 = x_483.flatten(2)
        x_483 = None
        x_484 = flatten_46.transpose(1, 2)
        flatten_46 = None
        x_485 = torch.nn.functional.layer_norm(
            x_484,
            (768,),
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_norm1_parameters_bias_ = (None)
        linear_92 = torch._C._nn.linear(
            x_485,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_bias_,
        )
        x_485 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_46 = linear_92.reshape(1, 49, 3, 24, 32)
        linear_92 = None
        qkv_23 = reshape_46.permute(2, 0, 3, 1, 4)
        reshape_46 = None
        unbind_23 = qkv_23.unbind(0)
        qkv_23 = None
        q_23 = unbind_23[0]
        k_34 = unbind_23[1]
        v_23 = unbind_23[2]
        unbind_23 = None
        k_35 = k_34 * 0.1767766952966369
        k_34 = None
        transpose_149 = k_35.transpose(-1, -2)
        k_35 = None
        attn_22 = transpose_149 @ v_23
        transpose_149 = v_23 = None
        attn_23 = attn_22.softmax(dim=-1)
        attn_22 = None
        transpose_150 = q_23.transpose(-1, -2)
        q_23 = None
        matmul_23 = attn_23 @ transpose_150
        attn_23 = transpose_150 = None
        x_486 = matmul_23.transpose(-1, -2)
        matmul_23 = None
        transpose_152 = x_486.transpose(1, 2)
        x_486 = None
        x_487 = transpose_152.reshape(1, 49, 768)
        transpose_152 = None
        x_488 = torch._C._nn.linear(
            x_487,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_487 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_489 = x_484 + x_488
        x_484 = x_488 = None
        transpose_153 = x_489.transpose(1, 2)
        x_489 = None
        view_142 = transpose_153.view(1, 768, 7, 7)
        transpose_153 = None
        feat_47 = torch.conv2d(
            view_142,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_bias_ = (None)
        x_490 = view_142 + feat_47
        view_142 = feat_47 = None
        flatten_47 = x_490.flatten(2)
        x_490 = None
        x_491 = flatten_47.transpose(1, 2)
        flatten_47 = None
        x_492 = torch.nn.functional.layer_norm(
            x_491,
            (768,),
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_norm2_parameters_bias_ = (None)
        x_493 = torch._C._nn.linear(
            x_492,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_492 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_494 = torch._C._nn.gelu(x_493, approximate="none")
        x_493 = None
        x_495 = torch.nn.functional.dropout(x_494, 0.0, False, False)
        x_494 = None
        x_496 = torch._C._nn.linear(
            x_495,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_495 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_497 = torch.nn.functional.dropout(x_496, 0.0, False, False)
        x_496 = None
        x_498 = x_491 + x_497
        x_491 = x_497 = None
        transpose_155 = x_498.transpose(1, 2)
        x_498 = None
        x_499 = transpose_155.view(1, 768, 7, 7)
        transpose_155 = None
        x_500 = torch.nn.functional.adaptive_avg_pool2d(x_499, 1)
        x_499 = None
        x_501 = x_500.permute(0, 2, 3, 1)
        x_500 = None
        x_502 = torch.nn.functional.layer_norm(
            x_501,
            (768,),
            l_self_modules_head_modules_norm_parameters_weight_,
            l_self_modules_head_modules_norm_parameters_bias_,
            1e-05,
        )
        x_501 = (
            l_self_modules_head_modules_norm_parameters_weight_
        ) = l_self_modules_head_modules_norm_parameters_bias_ = None
        x_503 = x_502.permute(0, 3, 1, 2)
        x_502 = None
        x_504 = x_503.flatten(1, -1)
        x_503 = None
        x_505 = torch.nn.functional.dropout(x_504, 0.0, False, False)
        x_504 = None
        x_506 = torch._C._nn.linear(
            x_505,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_505 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_506,)
