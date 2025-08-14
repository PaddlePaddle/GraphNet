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
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_1x1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_1x1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_pre_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_pre_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_pre_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_pre_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_kxk_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_1x1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_1x1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_1x1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_1x1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_kxk_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_1x1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_rel_pos_buffers_rel_coords_log_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_1x1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_1x1_parameters_bias_
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
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_1x1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_1x1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_pre_norm_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_pre_norm_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_pre_norm_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_pre_norm_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_pre_norm_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_pre_norm_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_pre_norm_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_pre_norm_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_1x1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_1x1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_kxk_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_kxk_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_se_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_se_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_se_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_se_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_se_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_se_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_se_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_se_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_1x1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_1x1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_1x1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_1x1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_
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
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_1x1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_1x1_parameters_bias_
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
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_1x1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_1x1_parameters_bias_
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
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_1x1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_1x1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_1x1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_1x1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_kxk_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_kxk_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_1x1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_1x1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_1x1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_1x1_parameters_bias_
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
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_rel_pos_buffers_rel_coords_log_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_rel_pos_buffers_rel_coords_log_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_
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
        x_5 = torch.nn.functional.batch_norm(
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
        x_6 = torch.conv2d(
            x_5,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_5 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_1x1_parameters_weight_ = (None)
        x_7 = torch.nn.functional.batch_norm(
            x_6,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_6 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (None)
        x_8 = torch.nn.functional.silu(x_7, inplace=True)
        x_7 = None
        x_9 = torch.conv2d(
            x_8,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_kxk_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            256,
        )
        x_8 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_kxk_parameters_weight_ = (None)
        x_10 = torch.nn.functional.batch_norm(
            x_9,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_9 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_11 = torch.nn.functional.silu(x_10, inplace=True)
        x_10 = None
        x_se = x_11.mean((2, 3), keepdim=True)
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
        x_se_2 = torch.nn.functional.silu(x_se_1, inplace=True)
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
        x_12 = x_11 * sigmoid
        x_11 = sigmoid = None
        x_13 = torch.conv2d(
            x_12,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_12 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_1x1_parameters_bias_ = (None)
        x_14 = x_13 + x_4
        x_13 = x_4 = None
        x_15 = torch.nn.functional.batch_norm(
            x_14,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_parameters_bias_ = (None)
        x_16 = torch.conv2d(
            x_15,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_15 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_1x1_parameters_weight_ = (None)
        x_17 = torch.nn.functional.batch_norm(
            x_16,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_16 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_18 = torch.nn.functional.silu(x_17, inplace=True)
        x_17 = None
        x_19 = torch.conv2d(
            x_18,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        x_18 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_kxk_parameters_weight_ = (None)
        x_20 = torch.nn.functional.batch_norm(
            x_19,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_19 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_21 = torch.nn.functional.silu(x_20, inplace=True)
        x_20 = None
        x_se_4 = x_21.mean((2, 3), keepdim=True)
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
        x_se_6 = torch.nn.functional.silu(x_se_5, inplace=True)
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
        x_22 = x_21 * sigmoid_1
        x_21 = sigmoid_1 = None
        x_23 = torch.conv2d(
            x_22,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_22 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_1x1_parameters_bias_ = (None)
        x_24 = x_23 + x_14
        x_23 = x_14 = None
        x_25 = torch.nn.functional.batch_norm(
            x_24,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_pre_norm_parameters_bias_ = (None)
        x_26 = torch.conv2d(
            x_25,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_25 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_1x1_parameters_weight_ = (None)
        x_27 = torch.nn.functional.batch_norm(
            x_26,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_26 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_bias_ = (None)
        x_28 = torch.nn.functional.silu(x_27, inplace=True)
        x_27 = None
        x_29 = torch.conv2d(
            x_28,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        x_28 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_kxk_parameters_weight_ = (None)
        x_30 = torch.nn.functional.batch_norm(
            x_29,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_29 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_bias_ = (None)
        x_31 = torch.nn.functional.silu(x_30, inplace=True)
        x_30 = None
        x_se_8 = x_31.mean((2, 3), keepdim=True)
        x_se_9 = torch.conv2d(
            x_se_8,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_8 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_se_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_se_modules_fc1_parameters_bias_ = (None)
        x_se_10 = torch.nn.functional.silu(x_se_9, inplace=True)
        x_se_9 = None
        x_se_11 = torch.conv2d(
            x_se_10,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_10 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_se_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_se_modules_fc2_parameters_bias_ = (None)
        sigmoid_2 = x_se_11.sigmoid()
        x_se_11 = None
        x_32 = x_31 * sigmoid_2
        x_31 = sigmoid_2 = None
        x_33 = torch.conv2d(
            x_32,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_32 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_1x1_parameters_bias_ = (None)
        x_34 = x_33 + x_24
        x_33 = x_24 = None
        x_35 = torch._C._nn.avg_pool2d(x_34, 2, 2, 0, False, True, None)
        x_36 = torch.conv2d(
            x_35,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_35 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_ = (None)
        x_37 = torch.nn.functional.batch_norm(
            x_34,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_34 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_parameters_bias_ = (None)
        x_38 = torch.conv2d(
            x_37,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_37 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_1x1_parameters_weight_ = (None)
        x_39 = torch.nn.functional.batch_norm(
            x_38,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_38 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (None)
        x_40 = torch.nn.functional.silu(x_39, inplace=True)
        x_39 = None
        x_41 = torch.conv2d(
            x_40,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_kxk_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            256,
        )
        x_40 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_kxk_parameters_weight_ = (None)
        x_42 = torch.nn.functional.batch_norm(
            x_41,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_41 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_43 = torch.nn.functional.silu(x_42, inplace=True)
        x_42 = None
        x_se_12 = x_43.mean((2, 3), keepdim=True)
        x_se_13 = torch.conv2d(
            x_se_12,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_12 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc1_parameters_bias_ = (None)
        x_se_14 = torch.nn.functional.silu(x_se_13, inplace=True)
        x_se_13 = None
        x_se_15 = torch.conv2d(
            x_se_14,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_14 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_se_modules_fc2_parameters_bias_ = (None)
        sigmoid_3 = x_se_15.sigmoid()
        x_se_15 = None
        x_44 = x_43 * sigmoid_3
        x_43 = sigmoid_3 = None
        x_45 = torch.conv2d(
            x_44,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_44 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_1x1_parameters_bias_ = (None)
        x_46 = x_45 + x_36
        x_45 = x_36 = None
        x_47 = torch.nn.functional.batch_norm(
            x_46,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_parameters_bias_ = (None)
        x_48 = torch.conv2d(
            x_47,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_47 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_1x1_parameters_weight_ = (None)
        x_49 = torch.nn.functional.batch_norm(
            x_48,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_48 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_50 = torch.nn.functional.silu(x_49, inplace=True)
        x_49 = None
        x_51 = torch.conv2d(
            x_50,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        x_50 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_kxk_parameters_weight_ = (None)
        x_52 = torch.nn.functional.batch_norm(
            x_51,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_51 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_53 = torch.nn.functional.silu(x_52, inplace=True)
        x_52 = None
        x_se_16 = x_53.mean((2, 3), keepdim=True)
        x_se_17 = torch.conv2d(
            x_se_16,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_16 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc1_parameters_bias_ = (None)
        x_se_18 = torch.nn.functional.silu(x_se_17, inplace=True)
        x_se_17 = None
        x_se_19 = torch.conv2d(
            x_se_18,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_18 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_se_modules_fc2_parameters_bias_ = (None)
        sigmoid_4 = x_se_19.sigmoid()
        x_se_19 = None
        x_54 = x_53 * sigmoid_4
        x_53 = sigmoid_4 = None
        x_55 = torch.conv2d(
            x_54,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_54 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_1x1_parameters_bias_ = (None)
        x_56 = x_55 + x_46
        x_55 = x_46 = None
        x_57 = torch.nn.functional.batch_norm(
            x_56,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_parameters_bias_ = (None)
        x_58 = torch.conv2d(
            x_57,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_57 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_1x1_parameters_weight_ = (None)
        x_59 = torch.nn.functional.batch_norm(
            x_58,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_58 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_bias_ = (None)
        x_60 = torch.nn.functional.silu(x_59, inplace=True)
        x_59 = None
        x_61 = torch.conv2d(
            x_60,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        x_60 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_kxk_parameters_weight_ = (None)
        x_62 = torch.nn.functional.batch_norm(
            x_61,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_61 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_bias_ = (None)
        x_63 = torch.nn.functional.silu(x_62, inplace=True)
        x_62 = None
        x_se_20 = x_63.mean((2, 3), keepdim=True)
        x_se_21 = torch.conv2d(
            x_se_20,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_20 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc1_parameters_bias_ = (None)
        x_se_22 = torch.nn.functional.silu(x_se_21, inplace=True)
        x_se_21 = None
        x_se_23 = torch.conv2d(
            x_se_22,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_22 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_se_modules_fc2_parameters_bias_ = (None)
        sigmoid_5 = x_se_23.sigmoid()
        x_se_23 = None
        x_64 = x_63 * sigmoid_5
        x_63 = sigmoid_5 = None
        x_65 = torch.conv2d(
            x_64,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_64 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_1x1_parameters_bias_ = (None)
        x_66 = x_65 + x_56
        x_65 = x_56 = None
        x_67 = torch.nn.functional.batch_norm(
            x_66,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_parameters_bias_ = (None)
        x_68 = torch.conv2d(
            x_67,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_67 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_1x1_parameters_weight_ = (None)
        x_69 = torch.nn.functional.batch_norm(
            x_68,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_68 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_bias_ = (None)
        x_70 = torch.nn.functional.silu(x_69, inplace=True)
        x_69 = None
        x_71 = torch.conv2d(
            x_70,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        x_70 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_kxk_parameters_weight_ = (None)
        x_72 = torch.nn.functional.batch_norm(
            x_71,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_71 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_bias_ = (None)
        x_73 = torch.nn.functional.silu(x_72, inplace=True)
        x_72 = None
        x_se_24 = x_73.mean((2, 3), keepdim=True)
        x_se_25 = torch.conv2d(
            x_se_24,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_24 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc1_parameters_bias_ = (None)
        x_se_26 = torch.nn.functional.silu(x_se_25, inplace=True)
        x_se_25 = None
        x_se_27 = torch.conv2d(
            x_se_26,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_26 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc2_parameters_bias_ = (None)
        sigmoid_6 = x_se_27.sigmoid()
        x_se_27 = None
        x_74 = x_73 * sigmoid_6
        x_73 = sigmoid_6 = None
        x_75 = torch.conv2d(
            x_74,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_74 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_1x1_parameters_bias_ = (None)
        x_76 = x_75 + x_66
        x_75 = x_66 = None
        x_77 = torch._C._nn.avg_pool2d(x_76, 2, 2, 0, False, True, None)
        x_78 = torch.conv2d(
            x_77,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_77 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_ = (None)
        x_79 = x_76.permute(0, 2, 3, 1)
        x_76 = None
        x_80 = torch.nn.functional.layer_norm(
            x_79,
            (128,),
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_,
            1e-06,
        )
        x_79 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_ = (None)
        x_81 = x_80.permute(0, 3, 1, 2)
        x_80 = None
        x_82 = torch._C._nn.avg_pool2d(x_81, 2, 2, 0, False, True, None)
        x_81 = None
        conv2d_39 = torch.conv2d(
            x_82,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_82 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        view = conv2d_39.view(1, 4, 96, -1)
        conv2d_39 = None
        chunk = view.chunk(3, dim=2)
        view = None
        q = chunk[0]
        k = chunk[1]
        v = chunk[2]
        chunk = None
        x_83 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_rel_coords_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_rel_coords_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_84 = torch.nn.functional.relu(x_83, inplace=False)
        x_83 = None
        x_85 = torch.nn.functional.dropout(x_84, 0.125, False, False)
        x_84 = None
        x_86 = torch._C._nn.linear(
            x_85,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_85 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_87 = torch.nn.functional.dropout(x_86, 0.0, False, False)
        x_86 = None
        view_1 = x_87.view(-1, 4)
        x_87 = None
        relative_position_bias = view_1[
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        view_1 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        relative_position_bias_1 = relative_position_bias.view((196, 196, 4))
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
        x_88 = transpose_3.reshape(1, -1, 14, 14)
        transpose_3 = None
        x_89 = torch.conv2d(
            x_88,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_88 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_90 = torch.nn.functional.dropout(x_89, 0.0, False, False)
        x_89 = None
        x_91 = x_78 + x_90
        x_78 = x_90 = None
        x_92 = x_91.permute(0, 2, 3, 1)
        x_93 = torch.nn.functional.layer_norm(
            x_92,
            (256,),
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_92 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_94 = x_93.permute(0, 3, 1, 2)
        x_93 = None
        x_95 = torch.conv2d(
            x_94,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_94 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_96 = torch._C._nn.gelu(x_95)
        x_95 = None
        x_97 = torch.nn.functional.dropout(x_96, 0.0, False, False)
        x_96 = None
        x_98 = torch.conv2d(
            x_97,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_97 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_99 = x_91 + x_98
        x_91 = x_98 = None
        x_100 = x_99.permute(0, 2, 3, 1)
        x_101 = torch.nn.functional.layer_norm(
            x_100,
            (256,),
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_100 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_102 = x_101.permute(0, 3, 1, 2)
        x_101 = None
        conv2d_43 = torch.conv2d(
            x_102,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_102 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_3 = conv2d_43.view(1, 8, 96, -1)
        conv2d_43 = None
        chunk_1 = view_3.chunk(3, dim=2)
        view_3 = None
        q_1 = chunk_1[0]
        k_1 = chunk_1[1]
        v_1 = chunk_1[2]
        chunk_1 = None
        x_103 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_rel_coords_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_rel_coords_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_104 = torch.nn.functional.relu(x_103, inplace=False)
        x_103 = None
        x_105 = torch.nn.functional.dropout(x_104, 0.125, False, False)
        x_104 = None
        x_106 = torch._C._nn.linear(
            x_105,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_105 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_107 = torch.nn.functional.dropout(x_106, 0.0, False, False)
        x_106 = None
        view_4 = x_107.view(-1, 8)
        x_107 = None
        relative_position_bias_3 = view_4[
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        view_4 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        relative_position_bias_4 = relative_position_bias_3.view((196, 196, 8))
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
        x_108 = transpose_7.reshape(1, -1, 14, 14)
        transpose_7 = None
        x_109 = torch.conv2d(
            x_108,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_108 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_110 = torch.nn.functional.dropout(x_109, 0.0, False, False)
        x_109 = None
        x_111 = x_99 + x_110
        x_99 = x_110 = None
        x_112 = x_111.permute(0, 2, 3, 1)
        x_113 = torch.nn.functional.layer_norm(
            x_112,
            (256,),
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_112 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_114 = x_113.permute(0, 3, 1, 2)
        x_113 = None
        x_115 = torch.conv2d(
            x_114,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_114 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_116 = torch._C._nn.gelu(x_115)
        x_115 = None
        x_117 = torch.nn.functional.dropout(x_116, 0.0, False, False)
        x_116 = None
        x_118 = torch.conv2d(
            x_117,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_117 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_119 = x_111 + x_118
        x_111 = x_118 = None
        x_120 = x_119.permute(0, 2, 3, 1)
        x_121 = torch.nn.functional.layer_norm(
            x_120,
            (256,),
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_120 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_ = (None)
        x_122 = x_121.permute(0, 3, 1, 2)
        x_121 = None
        conv2d_47 = torch.conv2d(
            x_122,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_122 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_6 = conv2d_47.view(1, 8, 96, -1)
        conv2d_47 = None
        chunk_2 = view_6.chunk(3, dim=2)
        view_6 = None
        q_2 = chunk_2[0]
        k_2 = chunk_2[1]
        v_2 = chunk_2[2]
        chunk_2 = None
        x_123 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_buffers_rel_coords_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_buffers_rel_coords_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_124 = torch.nn.functional.relu(x_123, inplace=False)
        x_123 = None
        x_125 = torch.nn.functional.dropout(x_124, 0.125, False, False)
        x_124 = None
        x_126 = torch._C._nn.linear(
            x_125,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_125 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_127 = torch.nn.functional.dropout(x_126, 0.0, False, False)
        x_126 = None
        view_7 = x_127.view(-1, 8)
        x_127 = None
        relative_position_bias_6 = view_7[
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        view_7 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        relative_position_bias_7 = relative_position_bias_6.view((196, 196, 8))
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
        x_128 = transpose_11.reshape(1, -1, 14, 14)
        transpose_11 = None
        x_129 = torch.conv2d(
            x_128,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_128 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_ = (None)
        x_130 = torch.nn.functional.dropout(x_129, 0.0, False, False)
        x_129 = None
        x_131 = x_119 + x_130
        x_119 = x_130 = None
        x_132 = x_131.permute(0, 2, 3, 1)
        x_133 = torch.nn.functional.layer_norm(
            x_132,
            (256,),
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_132 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_ = (None)
        x_134 = x_133.permute(0, 3, 1, 2)
        x_133 = None
        x_135 = torch.conv2d(
            x_134,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_134 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_136 = torch._C._nn.gelu(x_135)
        x_135 = None
        x_137 = torch.nn.functional.dropout(x_136, 0.0, False, False)
        x_136 = None
        x_138 = torch.conv2d(
            x_137,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_137 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_139 = x_131 + x_138
        x_131 = x_138 = None
        x_140 = x_139.permute(0, 2, 3, 1)
        x_141 = torch.nn.functional.layer_norm(
            x_140,
            (256,),
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_140 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_ = (None)
        x_142 = x_141.permute(0, 3, 1, 2)
        x_141 = None
        conv2d_51 = torch.conv2d(
            x_142,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_142 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_9 = conv2d_51.view(1, 8, 96, -1)
        conv2d_51 = None
        chunk_3 = view_9.chunk(3, dim=2)
        view_9 = None
        q_3 = chunk_3[0]
        k_3 = chunk_3[1]
        v_3 = chunk_3[2]
        chunk_3 = None
        x_143 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_buffers_rel_coords_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_buffers_rel_coords_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_144 = torch.nn.functional.relu(x_143, inplace=False)
        x_143 = None
        x_145 = torch.nn.functional.dropout(x_144, 0.125, False, False)
        x_144 = None
        x_146 = torch._C._nn.linear(
            x_145,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_145 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_147 = torch.nn.functional.dropout(x_146, 0.0, False, False)
        x_146 = None
        view_10 = x_147.view(-1, 8)
        x_147 = None
        relative_position_bias_9 = view_10[
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        view_10 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        relative_position_bias_10 = relative_position_bias_9.view((196, 196, 8))
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
        x_148 = transpose_15.reshape(1, -1, 14, 14)
        transpose_15 = None
        x_149 = torch.conv2d(
            x_148,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_148 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_ = (None)
        x_150 = torch.nn.functional.dropout(x_149, 0.0, False, False)
        x_149 = None
        x_151 = x_139 + x_150
        x_139 = x_150 = None
        x_152 = x_151.permute(0, 2, 3, 1)
        x_153 = torch.nn.functional.layer_norm(
            x_152,
            (256,),
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_152 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_ = (None)
        x_154 = x_153.permute(0, 3, 1, 2)
        x_153 = None
        x_155 = torch.conv2d(
            x_154,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_154 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_156 = torch._C._nn.gelu(x_155)
        x_155 = None
        x_157 = torch.nn.functional.dropout(x_156, 0.0, False, False)
        x_156 = None
        x_158 = torch.conv2d(
            x_157,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_157 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_159 = x_151 + x_158
        x_151 = x_158 = None
        x_160 = x_159.permute(0, 2, 3, 1)
        x_161 = torch.nn.functional.layer_norm(
            x_160,
            (256,),
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_160 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_ = (None)
        x_162 = x_161.permute(0, 3, 1, 2)
        x_161 = None
        conv2d_55 = torch.conv2d(
            x_162,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_162 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_12 = conv2d_55.view(1, 8, 96, -1)
        conv2d_55 = None
        chunk_4 = view_12.chunk(3, dim=2)
        view_12 = None
        q_4 = chunk_4[0]
        k_4 = chunk_4[1]
        v_4 = chunk_4[2]
        chunk_4 = None
        x_163 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_buffers_rel_coords_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_buffers_rel_coords_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_164 = torch.nn.functional.relu(x_163, inplace=False)
        x_163 = None
        x_165 = torch.nn.functional.dropout(x_164, 0.125, False, False)
        x_164 = None
        x_166 = torch._C._nn.linear(
            x_165,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_165 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_167 = torch.nn.functional.dropout(x_166, 0.0, False, False)
        x_166 = None
        view_13 = x_167.view(-1, 8)
        x_167 = None
        relative_position_bias_12 = view_13[
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        view_13 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        relative_position_bias_13 = relative_position_bias_12.view((196, 196, 8))
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
        x_168 = transpose_19.reshape(1, -1, 14, 14)
        transpose_19 = None
        x_169 = torch.conv2d(
            x_168,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_168 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_ = (None)
        x_170 = torch.nn.functional.dropout(x_169, 0.0, False, False)
        x_169 = None
        x_171 = x_159 + x_170
        x_159 = x_170 = None
        x_172 = x_171.permute(0, 2, 3, 1)
        x_173 = torch.nn.functional.layer_norm(
            x_172,
            (256,),
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_172 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_ = (None)
        x_174 = x_173.permute(0, 3, 1, 2)
        x_173 = None
        x_175 = torch.conv2d(
            x_174,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_174 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_176 = torch._C._nn.gelu(x_175)
        x_175 = None
        x_177 = torch.nn.functional.dropout(x_176, 0.0, False, False)
        x_176 = None
        x_178 = torch.conv2d(
            x_177,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_177 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_179 = x_171 + x_178
        x_171 = x_178 = None
        x_180 = x_179.permute(0, 2, 3, 1)
        x_181 = torch.nn.functional.layer_norm(
            x_180,
            (256,),
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_180 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_ = (None)
        x_182 = x_181.permute(0, 3, 1, 2)
        x_181 = None
        conv2d_59 = torch.conv2d(
            x_182,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_182 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_15 = conv2d_59.view(1, 8, 96, -1)
        conv2d_59 = None
        chunk_5 = view_15.chunk(3, dim=2)
        view_15 = None
        q_5 = chunk_5[0]
        k_5 = chunk_5[1]
        v_5 = chunk_5[2]
        chunk_5 = None
        x_183 = torch._C._nn.linear(
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_buffers_rel_coords_log_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_buffers_rel_coords_log_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_184 = torch.nn.functional.relu(x_183, inplace=False)
        x_183 = None
        x_185 = torch.nn.functional.dropout(x_184, 0.125, False, False)
        x_184 = None
        x_186 = torch._C._nn.linear(
            x_185,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_185 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_187 = torch.nn.functional.dropout(x_186, 0.0, False, False)
        x_186 = None
        view_16 = x_187.view(-1, 8)
        x_187 = None
        relative_position_bias_15 = view_16[
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        view_16 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        relative_position_bias_16 = relative_position_bias_15.view((196, 196, 8))
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
        x_188 = transpose_23.reshape(1, -1, 14, 14)
        transpose_23 = None
        x_189 = torch.conv2d(
            x_188,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_188 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_ = (None)
        x_190 = torch.nn.functional.dropout(x_189, 0.0, False, False)
        x_189 = None
        x_191 = x_179 + x_190
        x_179 = x_190 = None
        x_192 = x_191.permute(0, 2, 3, 1)
        x_193 = torch.nn.functional.layer_norm(
            x_192,
            (256,),
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_192 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_ = (None)
        x_194 = x_193.permute(0, 3, 1, 2)
        x_193 = None
        x_195 = torch.conv2d(
            x_194,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_194 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_196 = torch._C._nn.gelu(x_195)
        x_195 = None
        x_197 = torch.nn.functional.dropout(x_196, 0.0, False, False)
        x_196 = None
        x_198 = torch.conv2d(
            x_197,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_197 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_199 = x_191 + x_198
        x_191 = x_198 = None
        x_200 = torch._C._nn.avg_pool2d(x_199, 2, 2, 0, False, True, None)
        x_201 = torch.conv2d(
            x_200,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_200 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_ = (None)
        x_202 = x_199.permute(0, 2, 3, 1)
        x_199 = None
        x_203 = torch.nn.functional.layer_norm(
            x_202,
            (256,),
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_,
            1e-06,
        )
        x_202 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_ = (None)
        x_204 = x_203.permute(0, 3, 1, 2)
        x_203 = None
        x_205 = torch._C._nn.avg_pool2d(x_204, 2, 2, 0, False, True, None)
        x_204 = None
        conv2d_64 = torch.conv2d(
            x_205,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_205 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_18 = conv2d_64.view(1, 8, 96, -1)
        conv2d_64 = None
        chunk_6 = view_18.chunk(3, dim=2)
        view_18 = None
        q_6 = chunk_6[0]
        k_6 = chunk_6[1]
        v_6 = chunk_6[2]
        chunk_6 = None
        x_206 = torch._C._nn.linear(
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_rel_coords_log_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_rel_coords_log_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_207 = torch.nn.functional.relu(x_206, inplace=False)
        x_206 = None
        x_208 = torch.nn.functional.dropout(x_207, 0.125, False, False)
        x_207 = None
        x_209 = torch._C._nn.linear(
            x_208,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_208 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_210 = torch.nn.functional.dropout(x_209, 0.0, False, False)
        x_209 = None
        view_19 = x_210.view(-1, 8)
        x_210 = None
        relative_position_bias_18 = view_19[
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        view_19 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        relative_position_bias_19 = relative_position_bias_18.view((49, 49, 8))
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
        x_211 = transpose_27.reshape(1, -1, 7, 7)
        transpose_27 = None
        x_212 = torch.conv2d(
            x_211,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_211 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_213 = torch.nn.functional.dropout(x_212, 0.0, False, False)
        x_212 = None
        x_214 = x_201 + x_213
        x_201 = x_213 = None
        x_215 = x_214.permute(0, 2, 3, 1)
        x_216 = torch.nn.functional.layer_norm(
            x_215,
            (512,),
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_215 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_217 = x_216.permute(0, 3, 1, 2)
        x_216 = None
        x_218 = torch.conv2d(
            x_217,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_217 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_219 = torch._C._nn.gelu(x_218)
        x_218 = None
        x_220 = torch.nn.functional.dropout(x_219, 0.0, False, False)
        x_219 = None
        x_221 = torch.conv2d(
            x_220,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_220 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_222 = x_214 + x_221
        x_214 = x_221 = None
        x_223 = x_222.permute(0, 2, 3, 1)
        x_224 = torch.nn.functional.layer_norm(
            x_223,
            (512,),
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_223 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_225 = x_224.permute(0, 3, 1, 2)
        x_224 = None
        conv2d_68 = torch.conv2d(
            x_225,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_225 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_21 = conv2d_68.view(1, 16, 96, -1)
        conv2d_68 = None
        chunk_7 = view_21.chunk(3, dim=2)
        view_21 = None
        q_7 = chunk_7[0]
        k_7 = chunk_7[1]
        v_7 = chunk_7[2]
        chunk_7 = None
        x_226 = torch._C._nn.linear(
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_rel_coords_log_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_rel_coords_log_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_227 = torch.nn.functional.relu(x_226, inplace=False)
        x_226 = None
        x_228 = torch.nn.functional.dropout(x_227, 0.125, False, False)
        x_227 = None
        x_229 = torch._C._nn.linear(
            x_228,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_228 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_230 = torch.nn.functional.dropout(x_229, 0.0, False, False)
        x_229 = None
        view_22 = x_230.view(-1, 16)
        x_230 = None
        relative_position_bias_21 = view_22[
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        view_22 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        relative_position_bias_22 = relative_position_bias_21.view((49, 49, 16))
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
        x_231 = transpose_31.reshape(1, -1, 7, 7)
        transpose_31 = None
        x_232 = torch.conv2d(
            x_231,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_231 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_233 = torch.nn.functional.dropout(x_232, 0.0, False, False)
        x_232 = None
        x_234 = x_222 + x_233
        x_222 = x_233 = None
        x_235 = x_234.permute(0, 2, 3, 1)
        x_236 = torch.nn.functional.layer_norm(
            x_235,
            (512,),
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_235 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_237 = x_236.permute(0, 3, 1, 2)
        x_236 = None
        x_238 = torch.conv2d(
            x_237,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_237 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_239 = torch._C._nn.gelu(x_238)
        x_238 = None
        x_240 = torch.nn.functional.dropout(x_239, 0.0, False, False)
        x_239 = None
        x_241 = torch.conv2d(
            x_240,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_240 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_242 = x_234 + x_241
        x_234 = x_241 = None
        x_243 = x_242.permute(0, 2, 3, 1)
        x_244 = torch.nn.functional.layer_norm(
            x_243,
            (512,),
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_243 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_bias_ = (None)
        x_245 = x_244.permute(0, 3, 1, 2)
        x_244 = None
        conv2d_72 = torch.conv2d(
            x_245,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_245 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_24 = conv2d_72.view(1, 16, 96, -1)
        conv2d_72 = None
        chunk_8 = view_24.chunk(3, dim=2)
        view_24 = None
        q_8 = chunk_8[0]
        k_8 = chunk_8[1]
        v_8 = chunk_8[2]
        chunk_8 = None
        x_246 = torch._C._nn.linear(
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_rel_pos_buffers_rel_coords_log_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_rel_pos_buffers_rel_coords_log_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_247 = torch.nn.functional.relu(x_246, inplace=False)
        x_246 = None
        x_248 = torch.nn.functional.dropout(x_247, 0.125, False, False)
        x_247 = None
        x_249 = torch._C._nn.linear(
            x_248,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_248 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_rel_pos_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_250 = torch.nn.functional.dropout(x_249, 0.0, False, False)
        x_249 = None
        view_25 = x_250.view(-1, 16)
        x_250 = None
        relative_position_bias_24 = view_25[
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        view_25 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        relative_position_bias_25 = relative_position_bias_24.view((49, 49, 16))
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
        x_251 = transpose_35.reshape(1, -1, 7, 7)
        transpose_35 = None
        x_252 = torch.conv2d(
            x_251,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_251 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_ = (None)
        x_253 = torch.nn.functional.dropout(x_252, 0.0, False, False)
        x_252 = None
        x_254 = x_242 + x_253
        x_242 = x_253 = None
        x_255 = x_254.permute(0, 2, 3, 1)
        x_256 = torch.nn.functional.layer_norm(
            x_255,
            (512,),
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_255 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_bias_ = (None)
        x_257 = x_256.permute(0, 3, 1, 2)
        x_256 = None
        x_258 = torch.conv2d(
            x_257,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_257 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_259 = torch._C._nn.gelu(x_258)
        x_258 = None
        x_260 = torch.nn.functional.dropout(x_259, 0.0, False, False)
        x_259 = None
        x_261 = torch.conv2d(
            x_260,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_260 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_262 = x_254 + x_261
        x_254 = x_261 = None
        x_263 = x_262.permute(0, 2, 3, 1)
        x_262 = None
        x_264 = torch.nn.functional.layer_norm(
            x_263,
            (512,),
            l_self_modules_norm_parameters_weight_,
            l_self_modules_norm_parameters_bias_,
            1e-06,
        )
        x_263 = (
            l_self_modules_norm_parameters_weight_
        ) = l_self_modules_norm_parameters_bias_ = None
        x_265 = x_264.permute(0, 3, 1, 2)
        x_264 = None
        x_266 = torch.nn.functional.adaptive_avg_pool2d(x_265, 1)
        x_265 = None
        x_267 = x_266.flatten(1, -1)
        x_266 = None
        x_268 = torch.nn.functional.dropout(x_267, 0.0, False, False)
        x_267 = None
        x_269 = torch._C._nn.linear(
            x_268,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_268 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_269,)
