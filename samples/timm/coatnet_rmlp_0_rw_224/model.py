import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stem_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_x_: torch.Tensor,
        L_self_modules_stem_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_kxk_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_kxk_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_kxk_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_kxk_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_kxk_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_rel_coords_log_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_rel_coords_log_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_buffers_rel_coords_log_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_buffers_rel_coords_log_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_buffers_rel_coords_log_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_buffers_rel_coords_log_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_buffers_rel_coords_log_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_rel_coords_log_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_rel_coords_log_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_stem_modules_conv1_parameters_weight_ = (
            L_self_modules_stem_modules_conv1_parameters_weight_
        )
        l_x_ = L_x_
        l_self_modules_stem_modules_norm1_buffers_running_mean_ = (
            L_self_modules_stem_modules_norm1_buffers_running_mean_
        )
        l_self_modules_stem_modules_norm1_buffers_running_var_ = (
            L_self_modules_stem_modules_norm1_buffers_running_var_
        )
        l_self_modules_stem_modules_norm1_parameters_weight_ = (
            L_self_modules_stem_modules_norm1_parameters_weight_
        )
        l_self_modules_stem_modules_norm1_parameters_bias_ = (
            L_self_modules_stem_modules_norm1_parameters_bias_
        )
        l_self_modules_stem_modules_conv2_parameters_weight_ = (
            L_self_modules_stem_modules_conv2_parameters_weight_
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_1x1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_1x1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_kxk_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_kxk_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_1x1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_1x1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_1x1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_1x1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_kxk_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_kxk_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_1x1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_1x1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_1x1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_1x1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_kxk_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_kxk_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_1x1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_1x1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_1x1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_1x1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_kxk_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_kxk_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_1x1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_1x1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_1x1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_1x1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_kxk_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_kxk_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_1x1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_1x1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_rel_coords_log_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_rel_coords_log_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_rel_coords_log_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_rel_coords_log_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_buffers_rel_coords_log_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_buffers_rel_coords_log_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_buffers_rel_coords_log_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_buffers_rel_coords_log_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_buffers_rel_coords_log_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_buffers_rel_coords_log_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_buffers_rel_coords_log_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_buffers_rel_coords_log_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_buffers_rel_coords_log_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_buffers_rel_coords_log_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_rel_coords_log_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_rel_coords_log_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_rel_coords_log_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_rel_coords_log_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_norm_parameters_weight_ = L_self_modules_norm_parameters_weight_
        l_self_modules_norm_parameters_bias_ = L_self_modules_norm_parameters_bias_
        l_self_modules_head_modules_fc_parameters_weight_ = (
            L_self_modules_head_modules_fc_parameters_weight_
        )
        l_self_modules_head_modules_fc_parameters_bias_ = (
            L_self_modules_head_modules_fc_parameters_bias_
        )
        x = torch.conv2d(
            l_x_,
            l_self_modules_stem_modules_conv1_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_x_ = l_self_modules_stem_modules_conv1_parameters_weight_ = None
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_stem_modules_norm1_buffers_running_mean_,
            l_self_modules_stem_modules_norm1_buffers_running_var_,
            l_self_modules_stem_modules_norm1_parameters_weight_,
            l_self_modules_stem_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x = (
            l_self_modules_stem_modules_norm1_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_norm1_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_norm1_parameters_weight_
        ) = l_self_modules_stem_modules_norm1_parameters_bias_ = None
        x_2 = torch.nn.functional.silu(x_1, inplace=True)
        x_1 = None
        x_3 = torch.conv2d(
            x_2,
            l_self_modules_stem_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_2 = l_self_modules_stem_modules_conv2_parameters_weight_ = None
        x_4 = torch._C._nn.avg_pool2d(x_3, 2, 2, 0, False, True, None)
        x_5 = torch.conv2d(
            x_4,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_4 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_ = (None)
        x_6 = torch.nn.functional.batch_norm(
            x_3,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_3 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_pre_norm_parameters_bias_ = (None)
        x_7 = torch.nn.functional.silu(x_6, inplace=True)
        x_6 = None
        x_8 = torch.conv2d(
            x_7,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_7 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_1x1_parameters_weight_ = (None)
        x_9 = torch.nn.functional.batch_norm(
            x_8,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_8 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (None)
        x_10 = torch.nn.functional.silu(x_9, inplace=True)
        x_9 = None
        x_11 = torch.conv2d(
            x_10,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_kxk_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            256,
        )
        x_10 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_kxk_parameters_weight_ = (None)
        x_12 = torch.nn.functional.batch_norm(
            x_11,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_11 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_13 = torch.nn.functional.silu(x_12, inplace=True)
        x_12 = None
        x_se = x_13.mean((2, 3), keepdim=True)
        x_se_1 = torch.conv2d(
            x_se,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc1_parameters_bias_ = (None)
        x_se_2 = torch.nn.functional.relu(x_se_1, inplace=True)
        x_se_1 = None
        x_se_3 = torch.conv2d(
            x_se_2,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_2 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_se_modules_fc2_parameters_bias_ = (None)
        sigmoid = x_se_3.sigmoid()
        x_se_3 = None
        x_14 = x_13 * sigmoid
        x_13 = sigmoid = None
        x_15 = torch.conv2d(
            x_14,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_14 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_1x1_parameters_weight_ = (None)
        x_16 = x_15 + x_5
        x_15 = x_5 = None
        x_17 = torch.nn.functional.batch_norm(
            x_16,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_parameters_bias_ = (None)
        x_18 = torch.nn.functional.silu(x_17, inplace=True)
        x_17 = None
        x_19 = torch.conv2d(
            x_18,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_18 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_1x1_parameters_weight_ = (None)
        x_20 = torch.nn.functional.batch_norm(
            x_19,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_19 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_21 = torch.nn.functional.silu(x_20, inplace=True)
        x_20 = None
        x_22 = torch.conv2d(
            x_21,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        x_21 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_kxk_parameters_weight_ = (None)
        x_23 = torch.nn.functional.batch_norm(
            x_22,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_22 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_24 = torch.nn.functional.silu(x_23, inplace=True)
        x_23 = None
        x_se_4 = x_24.mean((2, 3), keepdim=True)
        x_se_5 = torch.conv2d(
            x_se_4,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_4 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc1_parameters_bias_ = (None)
        x_se_6 = torch.nn.functional.relu(x_se_5, inplace=True)
        x_se_5 = None
        x_se_7 = torch.conv2d(
            x_se_6,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_6 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_se_modules_fc2_parameters_bias_ = (None)
        sigmoid_1 = x_se_7.sigmoid()
        x_se_7 = None
        x_25 = x_24 * sigmoid_1
        x_24 = sigmoid_1 = None
        x_26 = torch.conv2d(
            x_25,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_25 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_1x1_parameters_weight_ = (None)
        x_27 = x_26 + x_16
        x_26 = x_16 = None
        x_28 = torch._C._nn.avg_pool2d(x_27, 2, 2, 0, False, True, None)
        x_29 = torch.conv2d(
            x_28,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_28 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_ = (None)
        x_30 = torch.nn.functional.batch_norm(
            x_27,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_27 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_parameters_bias_ = (None)
        x_31 = torch.nn.functional.silu(x_30, inplace=True)
        x_30 = None
        x_32 = torch.conv2d(
            x_31,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_31 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_1x1_parameters_weight_ = (None)
        x_33 = torch.nn.functional.batch_norm(
            x_32,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_32 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (None)
        x_34 = torch.nn.functional.silu(x_33, inplace=True)
        x_33 = None
        x_35 = torch.conv2d(
            x_34,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_kxk_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            384,
        )
        x_34 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_kxk_parameters_weight_ = (None)
        x_36 = torch.nn.functional.batch_norm(
            x_35,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_35 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_37 = torch.nn.functional.silu(x_36, inplace=True)
        x_36 = None
        x_se_8 = x_37.mean((2, 3), keepdim=True)
        x_se_9 = torch.conv2d(
            x_se_8,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_8 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc1_parameters_bias_ = (None)
        x_se_10 = torch.nn.functional.relu(x_se_9, inplace=True)
        x_se_9 = None
        x_se_11 = torch.conv2d(
            x_se_10,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_10 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc2_parameters_bias_ = (None)
        sigmoid_2 = x_se_11.sigmoid()
        x_se_11 = None
        x_38 = x_37 * sigmoid_2
        x_37 = sigmoid_2 = None
        x_39 = torch.conv2d(
            x_38,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_38 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_1x1_parameters_weight_ = (None)
        x_40 = x_39 + x_29
        x_39 = x_29 = None
        x_41 = torch.nn.functional.batch_norm(
            x_40,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_parameters_bias_ = (None)
        x_42 = torch.nn.functional.silu(x_41, inplace=True)
        x_41 = None
        x_43 = torch.conv2d(
            x_42,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_42 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_1x1_parameters_weight_ = (None)
        x_44 = torch.nn.functional.batch_norm(
            x_43,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_43 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_45 = torch.nn.functional.silu(x_44, inplace=True)
        x_44 = None
        x_46 = torch.conv2d(
            x_45,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        x_45 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_kxk_parameters_weight_ = (None)
        x_47 = torch.nn.functional.batch_norm(
            x_46,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_46 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_48 = torch.nn.functional.silu(x_47, inplace=True)
        x_47 = None
        x_se_12 = x_48.mean((2, 3), keepdim=True)
        x_se_13 = torch.conv2d(
            x_se_12,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_12 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc1_parameters_bias_ = (None)
        x_se_14 = torch.nn.functional.relu(x_se_13, inplace=True)
        x_se_13 = None
        x_se_15 = torch.conv2d(
            x_se_14,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_14 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc2_parameters_bias_ = (None)
        sigmoid_3 = x_se_15.sigmoid()
        x_se_15 = None
        x_49 = x_48 * sigmoid_3
        x_48 = sigmoid_3 = None
        x_50 = torch.conv2d(
            x_49,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_49 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_1x1_parameters_weight_ = (None)
        x_51 = x_50 + x_40
        x_50 = x_40 = None
        x_52 = torch.nn.functional.batch_norm(
            x_51,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_parameters_bias_ = (None)
        x_53 = torch.nn.functional.silu(x_52, inplace=True)
        x_52 = None
        x_54 = torch.conv2d(
            x_53,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_53 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_1x1_parameters_weight_ = (None)
        x_55 = torch.nn.functional.batch_norm(
            x_54,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_54 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_bias_ = (None)
        x_56 = torch.nn.functional.silu(x_55, inplace=True)
        x_55 = None
        x_57 = torch.conv2d(
            x_56,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        x_56 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_kxk_parameters_weight_ = (None)
        x_58 = torch.nn.functional.batch_norm(
            x_57,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_57 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_bias_ = (None)
        x_59 = torch.nn.functional.silu(x_58, inplace=True)
        x_58 = None
        x_se_16 = x_59.mean((2, 3), keepdim=True)
        x_se_17 = torch.conv2d(
            x_se_16,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_16 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc1_parameters_bias_ = (None)
        x_se_18 = torch.nn.functional.relu(x_se_17, inplace=True)
        x_se_17 = None
        x_se_19 = torch.conv2d(
            x_se_18,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_18 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc2_parameters_bias_ = (None)
        sigmoid_4 = x_se_19.sigmoid()
        x_se_19 = None
        x_60 = x_59 * sigmoid_4
        x_59 = sigmoid_4 = None
        x_61 = torch.conv2d(
            x_60,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_60 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_1x1_parameters_weight_ = (None)
        x_62 = x_61 + x_51
        x_61 = x_51 = None
        x_63 = torch._C._nn.avg_pool2d(x_62, 2, 2, 0, False, True, None)
        x_64 = torch.conv2d(
            x_63,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_63 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_ = (None)
        x_65 = x_62.permute(0, 2, 3, 1)
        x_62 = None
        x_66 = torch.nn.functional.layer_norm(
            x_65,
            (192,),
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_,
            1e-06,
        )
        x_65 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_ = (None)
        x_67 = x_66.permute(0, 3, 1, 2)
        x_66 = None
        x_68 = torch._C._nn.avg_pool2d(x_67, 2, 2, 0, False, True, None)
        x_67 = None
        conv2d_30 = torch.conv2d(
            x_68,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_68 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        view = conv2d_30.view(1, 6, 96, -1)
        conv2d_30 = None
        chunk = view.chunk(3, dim=2)
        view = None
        q = chunk[0]
        k = chunk[1]
        v = chunk[2]
        chunk = None
        x_69 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_rel_coords_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_rel_coords_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_70 = torch.nn.functional.relu(x_69, inplace=False)
        x_69 = None
        x_71 = torch.nn.functional.dropout(x_70, 0.125, False, False)
        x_70 = None
        x_72 = torch._C._nn.linear(
            x_71,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_71 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_73 = torch.nn.functional.dropout(x_72, 0.0, False, False)
        x_72 = None
        view_1 = x_73.view(-1, 6)
        x_73 = None
        relative_position_bias = view_1[
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        view_1 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        relative_position_bias_1 = relative_position_bias.view((196, 196, 6))
        relative_position_bias = None
        relative_position_bias_2 = relative_position_bias_1.permute(2, 0, 1)
        relative_position_bias_1 = None
        unsqueeze = relative_position_bias_2.unsqueeze(0)
        relative_position_bias_2 = None
        attn_bias = unsqueeze.contiguous()
        unsqueeze = None
        transpose = q.transpose(-1, -2)
        q = None
        contiguous_1 = transpose.contiguous()
        transpose = None
        transpose_1 = k.transpose(-1, -2)
        k = None
        contiguous_2 = transpose_1.contiguous()
        transpose_1 = None
        transpose_2 = v.transpose(-1, -2)
        v = None
        contiguous_3 = transpose_2.contiguous()
        transpose_2 = None
        scaled_dot_product_attention = torch._C._nn.scaled_dot_product_attention(
            contiguous_1, contiguous_2, contiguous_3, attn_mask=attn_bias, dropout_p=0.0
        )
        contiguous_1 = contiguous_2 = contiguous_3 = attn_bias = None
        transpose_3 = scaled_dot_product_attention.transpose(-1, -2)
        scaled_dot_product_attention = None
        x_74 = transpose_3.reshape(1, -1, 14, 14)
        transpose_3 = None
        x_75 = torch.conv2d(
            x_74,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_74 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_76 = torch.nn.functional.dropout(x_75, 0.0, False, False)
        x_75 = None
        x_77 = x_64 + x_76
        x_64 = x_76 = None
        x_78 = x_77.permute(0, 2, 3, 1)
        x_79 = torch.nn.functional.layer_norm(
            x_78,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_78 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_80 = x_79.permute(0, 3, 1, 2)
        x_79 = None
        x_81 = torch.conv2d(
            x_80,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_80 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_82 = torch._C._nn.gelu(x_81)
        x_81 = None
        x_83 = torch.nn.functional.dropout(x_82, 0.0, False, False)
        x_82 = None
        x_84 = torch.conv2d(
            x_83,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_83 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_85 = x_77 + x_84
        x_77 = x_84 = None
        x_86 = x_85.permute(0, 2, 3, 1)
        x_87 = torch.nn.functional.layer_norm(
            x_86,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_86 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_88 = x_87.permute(0, 3, 1, 2)
        x_87 = None
        conv2d_34 = torch.conv2d(
            x_88,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_88 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_3 = conv2d_34.view(1, 12, 96, -1)
        conv2d_34 = None
        chunk_1 = view_3.chunk(3, dim=2)
        view_3 = None
        q_1 = chunk_1[0]
        k_1 = chunk_1[1]
        v_1 = chunk_1[2]
        chunk_1 = None
        x_89 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_rel_coords_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_rel_coords_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_90 = torch.nn.functional.relu(x_89, inplace=False)
        x_89 = None
        x_91 = torch.nn.functional.dropout(x_90, 0.125, False, False)
        x_90 = None
        x_92 = torch._C._nn.linear(
            x_91,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_91 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_93 = torch.nn.functional.dropout(x_92, 0.0, False, False)
        x_92 = None
        view_4 = x_93.view(-1, 12)
        x_93 = None
        relative_position_bias_3 = view_4[
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        view_4 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        relative_position_bias_4 = relative_position_bias_3.view((196, 196, 12))
        relative_position_bias_3 = None
        relative_position_bias_5 = relative_position_bias_4.permute(2, 0, 1)
        relative_position_bias_4 = None
        unsqueeze_1 = relative_position_bias_5.unsqueeze(0)
        relative_position_bias_5 = None
        attn_bias_1 = unsqueeze_1.contiguous()
        unsqueeze_1 = None
        transpose_4 = q_1.transpose(-1, -2)
        q_1 = None
        contiguous_5 = transpose_4.contiguous()
        transpose_4 = None
        transpose_5 = k_1.transpose(-1, -2)
        k_1 = None
        contiguous_6 = transpose_5.contiguous()
        transpose_5 = None
        transpose_6 = v_1.transpose(-1, -2)
        v_1 = None
        contiguous_7 = transpose_6.contiguous()
        transpose_6 = None
        scaled_dot_product_attention_1 = torch._C._nn.scaled_dot_product_attention(
            contiguous_5,
            contiguous_6,
            contiguous_7,
            attn_mask=attn_bias_1,
            dropout_p=0.0,
        )
        contiguous_5 = contiguous_6 = contiguous_7 = attn_bias_1 = None
        transpose_7 = scaled_dot_product_attention_1.transpose(-1, -2)
        scaled_dot_product_attention_1 = None
        x_94 = transpose_7.reshape(1, -1, 14, 14)
        transpose_7 = None
        x_95 = torch.conv2d(
            x_94,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_94 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_96 = torch.nn.functional.dropout(x_95, 0.0, False, False)
        x_95 = None
        x_97 = x_85 + x_96
        x_85 = x_96 = None
        x_98 = x_97.permute(0, 2, 3, 1)
        x_99 = torch.nn.functional.layer_norm(
            x_98,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_98 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_100 = x_99.permute(0, 3, 1, 2)
        x_99 = None
        x_101 = torch.conv2d(
            x_100,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_100 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_102 = torch._C._nn.gelu(x_101)
        x_101 = None
        x_103 = torch.nn.functional.dropout(x_102, 0.0, False, False)
        x_102 = None
        x_104 = torch.conv2d(
            x_103,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_103 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_105 = x_97 + x_104
        x_97 = x_104 = None
        x_106 = x_105.permute(0, 2, 3, 1)
        x_107 = torch.nn.functional.layer_norm(
            x_106,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_106 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_ = (None)
        x_108 = x_107.permute(0, 3, 1, 2)
        x_107 = None
        conv2d_38 = torch.conv2d(
            x_108,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_108 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_6 = conv2d_38.view(1, 12, 96, -1)
        conv2d_38 = None
        chunk_2 = view_6.chunk(3, dim=2)
        view_6 = None
        q_2 = chunk_2[0]
        k_2 = chunk_2[1]
        v_2 = chunk_2[2]
        chunk_2 = None
        x_109 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_buffers_rel_coords_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_buffers_rel_coords_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_110 = torch.nn.functional.relu(x_109, inplace=False)
        x_109 = None
        x_111 = torch.nn.functional.dropout(x_110, 0.125, False, False)
        x_110 = None
        x_112 = torch._C._nn.linear(
            x_111,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_111 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_113 = torch.nn.functional.dropout(x_112, 0.0, False, False)
        x_112 = None
        view_7 = x_113.view(-1, 12)
        x_113 = None
        relative_position_bias_6 = view_7[
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        view_7 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        relative_position_bias_7 = relative_position_bias_6.view((196, 196, 12))
        relative_position_bias_6 = None
        relative_position_bias_8 = relative_position_bias_7.permute(2, 0, 1)
        relative_position_bias_7 = None
        unsqueeze_2 = relative_position_bias_8.unsqueeze(0)
        relative_position_bias_8 = None
        attn_bias_2 = unsqueeze_2.contiguous()
        unsqueeze_2 = None
        transpose_8 = q_2.transpose(-1, -2)
        q_2 = None
        contiguous_9 = transpose_8.contiguous()
        transpose_8 = None
        transpose_9 = k_2.transpose(-1, -2)
        k_2 = None
        contiguous_10 = transpose_9.contiguous()
        transpose_9 = None
        transpose_10 = v_2.transpose(-1, -2)
        v_2 = None
        contiguous_11 = transpose_10.contiguous()
        transpose_10 = None
        scaled_dot_product_attention_2 = torch._C._nn.scaled_dot_product_attention(
            contiguous_9,
            contiguous_10,
            contiguous_11,
            attn_mask=attn_bias_2,
            dropout_p=0.0,
        )
        contiguous_9 = contiguous_10 = contiguous_11 = attn_bias_2 = None
        transpose_11 = scaled_dot_product_attention_2.transpose(-1, -2)
        scaled_dot_product_attention_2 = None
        x_114 = transpose_11.reshape(1, -1, 14, 14)
        transpose_11 = None
        x_115 = torch.conv2d(
            x_114,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_114 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_ = (None)
        x_116 = torch.nn.functional.dropout(x_115, 0.0, False, False)
        x_115 = None
        x_117 = x_105 + x_116
        x_105 = x_116 = None
        x_118 = x_117.permute(0, 2, 3, 1)
        x_119 = torch.nn.functional.layer_norm(
            x_118,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_118 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_ = (None)
        x_120 = x_119.permute(0, 3, 1, 2)
        x_119 = None
        x_121 = torch.conv2d(
            x_120,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_120 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_122 = torch._C._nn.gelu(x_121)
        x_121 = None
        x_123 = torch.nn.functional.dropout(x_122, 0.0, False, False)
        x_122 = None
        x_124 = torch.conv2d(
            x_123,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_123 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_125 = x_117 + x_124
        x_117 = x_124 = None
        x_126 = x_125.permute(0, 2, 3, 1)
        x_127 = torch.nn.functional.layer_norm(
            x_126,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_126 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_ = (None)
        x_128 = x_127.permute(0, 3, 1, 2)
        x_127 = None
        conv2d_42 = torch.conv2d(
            x_128,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_128 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_9 = conv2d_42.view(1, 12, 96, -1)
        conv2d_42 = None
        chunk_3 = view_9.chunk(3, dim=2)
        view_9 = None
        q_3 = chunk_3[0]
        k_3 = chunk_3[1]
        v_3 = chunk_3[2]
        chunk_3 = None
        x_129 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_buffers_rel_coords_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_buffers_rel_coords_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_130 = torch.nn.functional.relu(x_129, inplace=False)
        x_129 = None
        x_131 = torch.nn.functional.dropout(x_130, 0.125, False, False)
        x_130 = None
        x_132 = torch._C._nn.linear(
            x_131,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_131 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_133 = torch.nn.functional.dropout(x_132, 0.0, False, False)
        x_132 = None
        view_10 = x_133.view(-1, 12)
        x_133 = None
        relative_position_bias_9 = view_10[
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        view_10 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        relative_position_bias_10 = relative_position_bias_9.view((196, 196, 12))
        relative_position_bias_9 = None
        relative_position_bias_11 = relative_position_bias_10.permute(2, 0, 1)
        relative_position_bias_10 = None
        unsqueeze_3 = relative_position_bias_11.unsqueeze(0)
        relative_position_bias_11 = None
        attn_bias_3 = unsqueeze_3.contiguous()
        unsqueeze_3 = None
        transpose_12 = q_3.transpose(-1, -2)
        q_3 = None
        contiguous_13 = transpose_12.contiguous()
        transpose_12 = None
        transpose_13 = k_3.transpose(-1, -2)
        k_3 = None
        contiguous_14 = transpose_13.contiguous()
        transpose_13 = None
        transpose_14 = v_3.transpose(-1, -2)
        v_3 = None
        contiguous_15 = transpose_14.contiguous()
        transpose_14 = None
        scaled_dot_product_attention_3 = torch._C._nn.scaled_dot_product_attention(
            contiguous_13,
            contiguous_14,
            contiguous_15,
            attn_mask=attn_bias_3,
            dropout_p=0.0,
        )
        contiguous_13 = contiguous_14 = contiguous_15 = attn_bias_3 = None
        transpose_15 = scaled_dot_product_attention_3.transpose(-1, -2)
        scaled_dot_product_attention_3 = None
        x_134 = transpose_15.reshape(1, -1, 14, 14)
        transpose_15 = None
        x_135 = torch.conv2d(
            x_134,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_134 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_ = (None)
        x_136 = torch.nn.functional.dropout(x_135, 0.0, False, False)
        x_135 = None
        x_137 = x_125 + x_136
        x_125 = x_136 = None
        x_138 = x_137.permute(0, 2, 3, 1)
        x_139 = torch.nn.functional.layer_norm(
            x_138,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_138 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_ = (None)
        x_140 = x_139.permute(0, 3, 1, 2)
        x_139 = None
        x_141 = torch.conv2d(
            x_140,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_140 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_142 = torch._C._nn.gelu(x_141)
        x_141 = None
        x_143 = torch.nn.functional.dropout(x_142, 0.0, False, False)
        x_142 = None
        x_144 = torch.conv2d(
            x_143,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_143 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_145 = x_137 + x_144
        x_137 = x_144 = None
        x_146 = x_145.permute(0, 2, 3, 1)
        x_147 = torch.nn.functional.layer_norm(
            x_146,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_146 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_ = (None)
        x_148 = x_147.permute(0, 3, 1, 2)
        x_147 = None
        conv2d_46 = torch.conv2d(
            x_148,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_148 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_12 = conv2d_46.view(1, 12, 96, -1)
        conv2d_46 = None
        chunk_4 = view_12.chunk(3, dim=2)
        view_12 = None
        q_4 = chunk_4[0]
        k_4 = chunk_4[1]
        v_4 = chunk_4[2]
        chunk_4 = None
        x_149 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_buffers_rel_coords_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_buffers_rel_coords_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_150 = torch.nn.functional.relu(x_149, inplace=False)
        x_149 = None
        x_151 = torch.nn.functional.dropout(x_150, 0.125, False, False)
        x_150 = None
        x_152 = torch._C._nn.linear(
            x_151,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_151 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_153 = torch.nn.functional.dropout(x_152, 0.0, False, False)
        x_152 = None
        view_13 = x_153.view(-1, 12)
        x_153 = None
        relative_position_bias_12 = view_13[
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        view_13 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        relative_position_bias_13 = relative_position_bias_12.view((196, 196, 12))
        relative_position_bias_12 = None
        relative_position_bias_14 = relative_position_bias_13.permute(2, 0, 1)
        relative_position_bias_13 = None
        unsqueeze_4 = relative_position_bias_14.unsqueeze(0)
        relative_position_bias_14 = None
        attn_bias_4 = unsqueeze_4.contiguous()
        unsqueeze_4 = None
        transpose_16 = q_4.transpose(-1, -2)
        q_4 = None
        contiguous_17 = transpose_16.contiguous()
        transpose_16 = None
        transpose_17 = k_4.transpose(-1, -2)
        k_4 = None
        contiguous_18 = transpose_17.contiguous()
        transpose_17 = None
        transpose_18 = v_4.transpose(-1, -2)
        v_4 = None
        contiguous_19 = transpose_18.contiguous()
        transpose_18 = None
        scaled_dot_product_attention_4 = torch._C._nn.scaled_dot_product_attention(
            contiguous_17,
            contiguous_18,
            contiguous_19,
            attn_mask=attn_bias_4,
            dropout_p=0.0,
        )
        contiguous_17 = contiguous_18 = contiguous_19 = attn_bias_4 = None
        transpose_19 = scaled_dot_product_attention_4.transpose(-1, -2)
        scaled_dot_product_attention_4 = None
        x_154 = transpose_19.reshape(1, -1, 14, 14)
        transpose_19 = None
        x_155 = torch.conv2d(
            x_154,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_154 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_ = (None)
        x_156 = torch.nn.functional.dropout(x_155, 0.0, False, False)
        x_155 = None
        x_157 = x_145 + x_156
        x_145 = x_156 = None
        x_158 = x_157.permute(0, 2, 3, 1)
        x_159 = torch.nn.functional.layer_norm(
            x_158,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_158 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_ = (None)
        x_160 = x_159.permute(0, 3, 1, 2)
        x_159 = None
        x_161 = torch.conv2d(
            x_160,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_160 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_162 = torch._C._nn.gelu(x_161)
        x_161 = None
        x_163 = torch.nn.functional.dropout(x_162, 0.0, False, False)
        x_162 = None
        x_164 = torch.conv2d(
            x_163,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_163 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_165 = x_157 + x_164
        x_157 = x_164 = None
        x_166 = x_165.permute(0, 2, 3, 1)
        x_167 = torch.nn.functional.layer_norm(
            x_166,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_166 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_ = (None)
        x_168 = x_167.permute(0, 3, 1, 2)
        x_167 = None
        conv2d_50 = torch.conv2d(
            x_168,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_168 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_15 = conv2d_50.view(1, 12, 96, -1)
        conv2d_50 = None
        chunk_5 = view_15.chunk(3, dim=2)
        view_15 = None
        q_5 = chunk_5[0]
        k_5 = chunk_5[1]
        v_5 = chunk_5[2]
        chunk_5 = None
        x_169 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_buffers_rel_coords_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_buffers_rel_coords_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_170 = torch.nn.functional.relu(x_169, inplace=False)
        x_169 = None
        x_171 = torch.nn.functional.dropout(x_170, 0.125, False, False)
        x_170 = None
        x_172 = torch._C._nn.linear(
            x_171,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_171 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_173 = torch.nn.functional.dropout(x_172, 0.0, False, False)
        x_172 = None
        view_16 = x_173.view(-1, 12)
        x_173 = None
        relative_position_bias_15 = view_16[
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        view_16 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        relative_position_bias_16 = relative_position_bias_15.view((196, 196, 12))
        relative_position_bias_15 = None
        relative_position_bias_17 = relative_position_bias_16.permute(2, 0, 1)
        relative_position_bias_16 = None
        unsqueeze_5 = relative_position_bias_17.unsqueeze(0)
        relative_position_bias_17 = None
        attn_bias_5 = unsqueeze_5.contiguous()
        unsqueeze_5 = None
        transpose_20 = q_5.transpose(-1, -2)
        q_5 = None
        contiguous_21 = transpose_20.contiguous()
        transpose_20 = None
        transpose_21 = k_5.transpose(-1, -2)
        k_5 = None
        contiguous_22 = transpose_21.contiguous()
        transpose_21 = None
        transpose_22 = v_5.transpose(-1, -2)
        v_5 = None
        contiguous_23 = transpose_22.contiguous()
        transpose_22 = None
        scaled_dot_product_attention_5 = torch._C._nn.scaled_dot_product_attention(
            contiguous_21,
            contiguous_22,
            contiguous_23,
            attn_mask=attn_bias_5,
            dropout_p=0.0,
        )
        contiguous_21 = contiguous_22 = contiguous_23 = attn_bias_5 = None
        transpose_23 = scaled_dot_product_attention_5.transpose(-1, -2)
        scaled_dot_product_attention_5 = None
        x_174 = transpose_23.reshape(1, -1, 14, 14)
        transpose_23 = None
        x_175 = torch.conv2d(
            x_174,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_174 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_ = (None)
        x_176 = torch.nn.functional.dropout(x_175, 0.0, False, False)
        x_175 = None
        x_177 = x_165 + x_176
        x_165 = x_176 = None
        x_178 = x_177.permute(0, 2, 3, 1)
        x_179 = torch.nn.functional.layer_norm(
            x_178,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_178 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_ = (None)
        x_180 = x_179.permute(0, 3, 1, 2)
        x_179 = None
        x_181 = torch.conv2d(
            x_180,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_180 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_182 = torch._C._nn.gelu(x_181)
        x_181 = None
        x_183 = torch.nn.functional.dropout(x_182, 0.0, False, False)
        x_182 = None
        x_184 = torch.conv2d(
            x_183,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_183 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_185 = x_177 + x_184
        x_177 = x_184 = None
        x_186 = x_185.permute(0, 2, 3, 1)
        x_187 = torch.nn.functional.layer_norm(
            x_186,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_186 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_bias_ = (None)
        x_188 = x_187.permute(0, 3, 1, 2)
        x_187 = None
        conv2d_54 = torch.conv2d(
            x_188,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_188 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_18 = conv2d_54.view(1, 12, 96, -1)
        conv2d_54 = None
        chunk_6 = view_18.chunk(3, dim=2)
        view_18 = None
        q_6 = chunk_6[0]
        k_6 = chunk_6[1]
        v_6 = chunk_6[2]
        chunk_6 = None
        x_189 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_buffers_rel_coords_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_buffers_rel_coords_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_190 = torch.nn.functional.relu(x_189, inplace=False)
        x_189 = None
        x_191 = torch.nn.functional.dropout(x_190, 0.125, False, False)
        x_190 = None
        x_192 = torch._C._nn.linear(
            x_191,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_191 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_193 = torch.nn.functional.dropout(x_192, 0.0, False, False)
        x_192 = None
        view_19 = x_193.view(-1, 12)
        x_193 = None
        relative_position_bias_18 = view_19[
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        view_19 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        relative_position_bias_19 = relative_position_bias_18.view((196, 196, 12))
        relative_position_bias_18 = None
        relative_position_bias_20 = relative_position_bias_19.permute(2, 0, 1)
        relative_position_bias_19 = None
        unsqueeze_6 = relative_position_bias_20.unsqueeze(0)
        relative_position_bias_20 = None
        attn_bias_6 = unsqueeze_6.contiguous()
        unsqueeze_6 = None
        transpose_24 = q_6.transpose(-1, -2)
        q_6 = None
        contiguous_25 = transpose_24.contiguous()
        transpose_24 = None
        transpose_25 = k_6.transpose(-1, -2)
        k_6 = None
        contiguous_26 = transpose_25.contiguous()
        transpose_25 = None
        transpose_26 = v_6.transpose(-1, -2)
        v_6 = None
        contiguous_27 = transpose_26.contiguous()
        transpose_26 = None
        scaled_dot_product_attention_6 = torch._C._nn.scaled_dot_product_attention(
            contiguous_25,
            contiguous_26,
            contiguous_27,
            attn_mask=attn_bias_6,
            dropout_p=0.0,
        )
        contiguous_25 = contiguous_26 = contiguous_27 = attn_bias_6 = None
        transpose_27 = scaled_dot_product_attention_6.transpose(-1, -2)
        scaled_dot_product_attention_6 = None
        x_194 = transpose_27.reshape(1, -1, 14, 14)
        transpose_27 = None
        x_195 = torch.conv2d(
            x_194,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_194 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_ = (None)
        x_196 = torch.nn.functional.dropout(x_195, 0.0, False, False)
        x_195 = None
        x_197 = x_185 + x_196
        x_185 = x_196 = None
        x_198 = x_197.permute(0, 2, 3, 1)
        x_199 = torch.nn.functional.layer_norm(
            x_198,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_198 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_bias_ = (None)
        x_200 = x_199.permute(0, 3, 1, 2)
        x_199 = None
        x_201 = torch.conv2d(
            x_200,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_200 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_202 = torch._C._nn.gelu(x_201)
        x_201 = None
        x_203 = torch.nn.functional.dropout(x_202, 0.0, False, False)
        x_202 = None
        x_204 = torch.conv2d(
            x_203,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_203 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_205 = x_197 + x_204
        x_197 = x_204 = None
        x_206 = torch._C._nn.avg_pool2d(x_205, 2, 2, 0, False, True, None)
        x_207 = torch.conv2d(
            x_206,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_206 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_ = (None)
        x_208 = x_205.permute(0, 2, 3, 1)
        x_205 = None
        x_209 = torch.nn.functional.layer_norm(
            x_208,
            (384,),
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_,
            1e-06,
        )
        x_208 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_ = (None)
        x_210 = x_209.permute(0, 3, 1, 2)
        x_209 = None
        x_211 = torch._C._nn.avg_pool2d(x_210, 2, 2, 0, False, True, None)
        x_210 = None
        conv2d_59 = torch.conv2d(
            x_211,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_211 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_21 = conv2d_59.view(1, 12, 96, -1)
        conv2d_59 = None
        chunk_7 = view_21.chunk(3, dim=2)
        view_21 = None
        q_7 = chunk_7[0]
        k_7 = chunk_7[1]
        v_7 = chunk_7[2]
        chunk_7 = None
        x_212 = torch._C._nn.linear(
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_rel_coords_log_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_rel_coords_log_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_213 = torch.nn.functional.relu(x_212, inplace=False)
        x_212 = None
        x_214 = torch.nn.functional.dropout(x_213, 0.125, False, False)
        x_213 = None
        x_215 = torch._C._nn.linear(
            x_214,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_214 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_216 = torch.nn.functional.dropout(x_215, 0.0, False, False)
        x_215 = None
        view_22 = x_216.view(-1, 12)
        x_216 = None
        relative_position_bias_21 = view_22[
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        view_22 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        relative_position_bias_22 = relative_position_bias_21.view((49, 49, 12))
        relative_position_bias_21 = None
        relative_position_bias_23 = relative_position_bias_22.permute(2, 0, 1)
        relative_position_bias_22 = None
        unsqueeze_7 = relative_position_bias_23.unsqueeze(0)
        relative_position_bias_23 = None
        attn_bias_7 = unsqueeze_7.contiguous()
        unsqueeze_7 = None
        transpose_28 = q_7.transpose(-1, -2)
        q_7 = None
        contiguous_29 = transpose_28.contiguous()
        transpose_28 = None
        transpose_29 = k_7.transpose(-1, -2)
        k_7 = None
        contiguous_30 = transpose_29.contiguous()
        transpose_29 = None
        transpose_30 = v_7.transpose(-1, -2)
        v_7 = None
        contiguous_31 = transpose_30.contiguous()
        transpose_30 = None
        scaled_dot_product_attention_7 = torch._C._nn.scaled_dot_product_attention(
            contiguous_29,
            contiguous_30,
            contiguous_31,
            attn_mask=attn_bias_7,
            dropout_p=0.0,
        )
        contiguous_29 = contiguous_30 = contiguous_31 = attn_bias_7 = None
        transpose_31 = scaled_dot_product_attention_7.transpose(-1, -2)
        scaled_dot_product_attention_7 = None
        x_217 = transpose_31.reshape(1, -1, 7, 7)
        transpose_31 = None
        x_218 = torch.conv2d(
            x_217,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_217 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_219 = torch.nn.functional.dropout(x_218, 0.0, False, False)
        x_218 = None
        x_220 = x_207 + x_219
        x_207 = x_219 = None
        x_221 = x_220.permute(0, 2, 3, 1)
        x_222 = torch.nn.functional.layer_norm(
            x_221,
            (768,),
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_221 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_223 = x_222.permute(0, 3, 1, 2)
        x_222 = None
        x_224 = torch.conv2d(
            x_223,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_223 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_225 = torch._C._nn.gelu(x_224)
        x_224 = None
        x_226 = torch.nn.functional.dropout(x_225, 0.0, False, False)
        x_225 = None
        x_227 = torch.conv2d(
            x_226,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_226 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_228 = x_220 + x_227
        x_220 = x_227 = None
        x_229 = x_228.permute(0, 2, 3, 1)
        x_230 = torch.nn.functional.layer_norm(
            x_229,
            (768,),
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_229 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_231 = x_230.permute(0, 3, 1, 2)
        x_230 = None
        conv2d_63 = torch.conv2d(
            x_231,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_231 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_24 = conv2d_63.view(1, 24, 96, -1)
        conv2d_63 = None
        chunk_8 = view_24.chunk(3, dim=2)
        view_24 = None
        q_8 = chunk_8[0]
        k_8 = chunk_8[1]
        v_8 = chunk_8[2]
        chunk_8 = None
        x_232 = torch._C._nn.linear(
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_rel_coords_log_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_rel_coords_log_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_233 = torch.nn.functional.relu(x_232, inplace=False)
        x_232 = None
        x_234 = torch.nn.functional.dropout(x_233, 0.125, False, False)
        x_233 = None
        x_235 = torch._C._nn.linear(
            x_234,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_234 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_236 = torch.nn.functional.dropout(x_235, 0.0, False, False)
        x_235 = None
        view_25 = x_236.view(-1, 24)
        x_236 = None
        relative_position_bias_24 = view_25[
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        view_25 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        relative_position_bias_25 = relative_position_bias_24.view((49, 49, 24))
        relative_position_bias_24 = None
        relative_position_bias_26 = relative_position_bias_25.permute(2, 0, 1)
        relative_position_bias_25 = None
        unsqueeze_8 = relative_position_bias_26.unsqueeze(0)
        relative_position_bias_26 = None
        attn_bias_8 = unsqueeze_8.contiguous()
        unsqueeze_8 = None
        transpose_32 = q_8.transpose(-1, -2)
        q_8 = None
        contiguous_33 = transpose_32.contiguous()
        transpose_32 = None
        transpose_33 = k_8.transpose(-1, -2)
        k_8 = None
        contiguous_34 = transpose_33.contiguous()
        transpose_33 = None
        transpose_34 = v_8.transpose(-1, -2)
        v_8 = None
        contiguous_35 = transpose_34.contiguous()
        transpose_34 = None
        scaled_dot_product_attention_8 = torch._C._nn.scaled_dot_product_attention(
            contiguous_33,
            contiguous_34,
            contiguous_35,
            attn_mask=attn_bias_8,
            dropout_p=0.0,
        )
        contiguous_33 = contiguous_34 = contiguous_35 = attn_bias_8 = None
        transpose_35 = scaled_dot_product_attention_8.transpose(-1, -2)
        scaled_dot_product_attention_8 = None
        x_237 = transpose_35.reshape(1, -1, 7, 7)
        transpose_35 = None
        x_238 = torch.conv2d(
            x_237,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_237 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_239 = torch.nn.functional.dropout(x_238, 0.0, False, False)
        x_238 = None
        x_240 = x_228 + x_239
        x_228 = x_239 = None
        x_241 = x_240.permute(0, 2, 3, 1)
        x_242 = torch.nn.functional.layer_norm(
            x_241,
            (768,),
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_241 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_243 = x_242.permute(0, 3, 1, 2)
        x_242 = None
        x_244 = torch.conv2d(
            x_243,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_243 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_245 = torch._C._nn.gelu(x_244)
        x_244 = None
        x_246 = torch.nn.functional.dropout(x_245, 0.0, False, False)
        x_245 = None
        x_247 = torch.conv2d(
            x_246,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_246 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_248 = x_240 + x_247
        x_240 = x_247 = None
        x_249 = x_248.permute(0, 2, 3, 1)
        x_248 = None
        x_250 = torch.nn.functional.layer_norm(
            x_249,
            (768,),
            l_self_modules_norm_parameters_weight_,
            l_self_modules_norm_parameters_bias_,
            1e-06,
        )
        x_249 = (
            l_self_modules_norm_parameters_weight_
        ) = l_self_modules_norm_parameters_bias_ = None
        x_251 = x_250.permute(0, 3, 1, 2)
        x_250 = None
        x_252 = torch.nn.functional.adaptive_avg_pool2d(x_251, 1)
        x_251 = None
        x_253 = x_252.flatten(1, -1)
        x_252 = None
        x_254 = torch.nn.functional.dropout(x_253, 0.0, False, False)
        x_253 = None
        x_255 = torch._C._nn.linear(
            x_254,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_254 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_255,)
