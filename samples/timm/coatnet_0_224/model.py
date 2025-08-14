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
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_pre_logits_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_pre_logits_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
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
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
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
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
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
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
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
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
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
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_head_modules_norm_parameters_weight_ = (
            L_self_modules_head_modules_norm_parameters_weight_
        )
        l_self_modules_head_modules_norm_parameters_bias_ = (
            L_self_modules_head_modules_norm_parameters_bias_
        )
        l_self_modules_head_modules_pre_logits_modules_fc_parameters_weight_ = (
            L_self_modules_head_modules_pre_logits_modules_fc_parameters_weight_
        )
        l_self_modules_head_modules_pre_logits_modules_fc_parameters_bias_ = (
            L_self_modules_head_modules_pre_logits_modules_fc_parameters_bias_
        )
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
        x_2 = torch._C._nn.gelu(x_1)
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
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_4 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_ = (None)
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
        x_7 = torch.conv2d(
            x_6,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_6 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_1x1_parameters_weight_ = (None)
        x_8 = torch.nn.functional.batch_norm(
            x_7,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_7 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (None)
        x_9 = torch._C._nn.gelu(x_8)
        x_8 = None
        x_10 = torch.conv2d(
            x_9,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_kxk_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            384,
        )
        x_9 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_kxk_parameters_weight_ = (None)
        x_11 = torch.nn.functional.batch_norm(
            x_10,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_10 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_12 = torch._C._nn.gelu(x_11)
        x_11 = None
        x_se = x_12.mean((2, 3), keepdim=True)
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
        x_13 = x_12 * sigmoid
        x_12 = sigmoid = None
        x_14 = torch.conv2d(
            x_13,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_13 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_1x1_parameters_bias_ = (None)
        x_15 = x_14 + x_5
        x_14 = x_5 = None
        x_16 = torch.nn.functional.batch_norm(
            x_15,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_pre_norm_parameters_bias_ = (None)
        x_17 = torch.conv2d(
            x_16,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_16 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_1x1_parameters_weight_ = (None)
        x_18 = torch.nn.functional.batch_norm(
            x_17,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_17 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_19 = torch._C._nn.gelu(x_18)
        x_18 = None
        x_20 = torch.conv2d(
            x_19,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        x_19 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_kxk_parameters_weight_ = (None)
        x_21 = torch.nn.functional.batch_norm(
            x_20,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_20 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_22 = torch._C._nn.gelu(x_21)
        x_21 = None
        x_se_4 = x_22.mean((2, 3), keepdim=True)
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
        x_23 = x_22 * sigmoid_1
        x_22 = sigmoid_1 = None
        x_24 = torch.conv2d(
            x_23,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_23 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_1x1_parameters_bias_ = (None)
        x_25 = x_24 + x_15
        x_24 = x_15 = None
        x_26 = torch._C._nn.avg_pool2d(x_25, 2, 2, 0, False, True, None)
        x_27 = torch.conv2d(
            x_26,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_26 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_ = (None)
        x_28 = torch.nn.functional.batch_norm(
            x_25,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_25 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_parameters_bias_ = (None)
        x_29 = torch.conv2d(
            x_28,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_28 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_1x1_parameters_weight_ = (None)
        x_30 = torch.nn.functional.batch_norm(
            x_29,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_29 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (None)
        x_31 = torch._C._nn.gelu(x_30)
        x_30 = None
        x_32 = torch.conv2d(
            x_31,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_kxk_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            768,
        )
        x_31 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_kxk_parameters_weight_ = (None)
        x_33 = torch.nn.functional.batch_norm(
            x_32,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_32 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_34 = torch._C._nn.gelu(x_33)
        x_33 = None
        x_se_8 = x_34.mean((2, 3), keepdim=True)
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
        x_se_10 = torch.nn.functional.silu(x_se_9, inplace=True)
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
        x_35 = x_34 * sigmoid_2
        x_34 = sigmoid_2 = None
        x_36 = torch.conv2d(
            x_35,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_35 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_1x1_parameters_bias_ = (None)
        x_37 = x_36 + x_27
        x_36 = x_27 = None
        x_38 = torch.nn.functional.batch_norm(
            x_37,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_parameters_bias_ = (None)
        x_39 = torch.conv2d(
            x_38,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_38 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_1x1_parameters_weight_ = (None)
        x_40 = torch.nn.functional.batch_norm(
            x_39,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_39 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_41 = torch._C._nn.gelu(x_40)
        x_40 = None
        x_42 = torch.conv2d(
            x_41,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        x_41 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_kxk_parameters_weight_ = (None)
        x_43 = torch.nn.functional.batch_norm(
            x_42,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_42 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_44 = torch._C._nn.gelu(x_43)
        x_43 = None
        x_se_12 = x_44.mean((2, 3), keepdim=True)
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
        x_se_14 = torch.nn.functional.silu(x_se_13, inplace=True)
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
        x_45 = x_44 * sigmoid_3
        x_44 = sigmoid_3 = None
        x_46 = torch.conv2d(
            x_45,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_45 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_1x1_parameters_bias_ = (None)
        x_47 = x_46 + x_37
        x_46 = x_37 = None
        x_48 = torch.nn.functional.batch_norm(
            x_47,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_parameters_bias_ = (None)
        x_49 = torch.conv2d(
            x_48,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_48 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_1x1_parameters_weight_ = (None)
        x_50 = torch.nn.functional.batch_norm(
            x_49,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_49 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_bias_ = (None)
        x_51 = torch._C._nn.gelu(x_50)
        x_50 = None
        x_52 = torch.conv2d(
            x_51,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        x_51 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_kxk_parameters_weight_ = (None)
        x_53 = torch.nn.functional.batch_norm(
            x_52,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_52 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_bias_ = (None)
        x_54 = torch._C._nn.gelu(x_53)
        x_53 = None
        x_se_16 = x_54.mean((2, 3), keepdim=True)
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
        x_se_18 = torch.nn.functional.silu(x_se_17, inplace=True)
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
        x_55 = x_54 * sigmoid_4
        x_54 = sigmoid_4 = None
        x_56 = torch.conv2d(
            x_55,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_1x1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_55 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_1x1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_1x1_parameters_bias_ = (None)
        x_57 = x_56 + x_47
        x_56 = x_47 = None
        x_58 = torch._C._nn.avg_pool2d(x_57, 2, 2, 0, False, True, None)
        x_59 = torch.conv2d(
            x_58,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_58 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_ = (None)
        x_60 = x_57.permute(0, 2, 3, 1)
        x_57 = None
        x_61 = torch.nn.functional.layer_norm(
            x_60,
            (192,),
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_,
            1e-06,
        )
        x_60 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_ = (None)
        x_62 = x_61.permute(0, 3, 1, 2)
        x_61 = None
        x_63 = torch._C._nn.avg_pool2d(x_62, 2, 2, 0, False, True, None)
        x_62 = None
        conv2d_30 = torch.conv2d(
            x_63,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_63 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        view = conv2d_30.view(1, 12, 96, -1)
        conv2d_30 = None
        chunk = view.chunk(3, dim=2)
        view = None
        q = chunk[0]
        k = chunk[1]
        v = chunk[2]
        chunk = None
        relative_position_bias = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_1 = relative_position_bias.view((196, 196, 12))
        relative_position_bias = None
        relative_position_bias_1 = view_1.permute(2, 0, 1)
        view_1 = None
        unsqueeze = relative_position_bias_1.unsqueeze(0)
        relative_position_bias_1 = None
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
        x_64 = transpose_3.reshape(1, -1, 14, 14)
        transpose_3 = None
        x_65 = torch.conv2d(
            x_64,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_64 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_66 = torch.nn.functional.dropout(x_65, 0.0, False, False)
        x_65 = None
        x_67 = x_59 + x_66
        x_59 = x_66 = None
        x_68 = x_67.permute(0, 2, 3, 1)
        x_69 = torch.nn.functional.layer_norm(
            x_68,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_68 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_70 = x_69.permute(0, 3, 1, 2)
        x_69 = None
        x_71 = torch.conv2d(
            x_70,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_70 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_72 = torch._C._nn.gelu(x_71)
        x_71 = None
        x_73 = torch.nn.functional.dropout(x_72, 0.0, False, False)
        x_72 = None
        x_74 = torch.conv2d(
            x_73,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_73 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_75 = x_67 + x_74
        x_67 = x_74 = None
        x_76 = x_75.permute(0, 2, 3, 1)
        x_77 = torch.nn.functional.layer_norm(
            x_76,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_76 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_78 = x_77.permute(0, 3, 1, 2)
        x_77 = None
        conv2d_34 = torch.conv2d(
            x_78,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_78 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_2 = conv2d_34.view(1, 12, 96, -1)
        conv2d_34 = None
        chunk_1 = view_2.chunk(3, dim=2)
        view_2 = None
        q_1 = chunk_1[0]
        k_1 = chunk_1[1]
        v_1 = chunk_1[2]
        chunk_1 = None
        relative_position_bias_2 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_3 = relative_position_bias_2.view((196, 196, 12))
        relative_position_bias_2 = None
        relative_position_bias_3 = view_3.permute(2, 0, 1)
        view_3 = None
        unsqueeze_1 = relative_position_bias_3.unsqueeze(0)
        relative_position_bias_3 = None
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
        x_79 = transpose_7.reshape(1, -1, 14, 14)
        transpose_7 = None
        x_80 = torch.conv2d(
            x_79,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_79 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_81 = torch.nn.functional.dropout(x_80, 0.0, False, False)
        x_80 = None
        x_82 = x_75 + x_81
        x_75 = x_81 = None
        x_83 = x_82.permute(0, 2, 3, 1)
        x_84 = torch.nn.functional.layer_norm(
            x_83,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_83 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_85 = x_84.permute(0, 3, 1, 2)
        x_84 = None
        x_86 = torch.conv2d(
            x_85,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_85 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_87 = torch._C._nn.gelu(x_86)
        x_86 = None
        x_88 = torch.nn.functional.dropout(x_87, 0.0, False, False)
        x_87 = None
        x_89 = torch.conv2d(
            x_88,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_88 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_90 = x_82 + x_89
        x_82 = x_89 = None
        x_91 = x_90.permute(0, 2, 3, 1)
        x_92 = torch.nn.functional.layer_norm(
            x_91,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_91 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_ = (None)
        x_93 = x_92.permute(0, 3, 1, 2)
        x_92 = None
        conv2d_38 = torch.conv2d(
            x_93,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_93 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_4 = conv2d_38.view(1, 12, 96, -1)
        conv2d_38 = None
        chunk_2 = view_4.chunk(3, dim=2)
        view_4 = None
        q_2 = chunk_2[0]
        k_2 = chunk_2[1]
        v_2 = chunk_2[2]
        chunk_2 = None
        relative_position_bias_4 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_5 = relative_position_bias_4.view((196, 196, 12))
        relative_position_bias_4 = None
        relative_position_bias_5 = view_5.permute(2, 0, 1)
        view_5 = None
        unsqueeze_2 = relative_position_bias_5.unsqueeze(0)
        relative_position_bias_5 = None
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
        x_94 = transpose_11.reshape(1, -1, 14, 14)
        transpose_11 = None
        x_95 = torch.conv2d(
            x_94,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_94 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_ = (None)
        x_96 = torch.nn.functional.dropout(x_95, 0.0, False, False)
        x_95 = None
        x_97 = x_90 + x_96
        x_90 = x_96 = None
        x_98 = x_97.permute(0, 2, 3, 1)
        x_99 = torch.nn.functional.layer_norm(
            x_98,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_98 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_ = (None)
        x_100 = x_99.permute(0, 3, 1, 2)
        x_99 = None
        x_101 = torch.conv2d(
            x_100,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_100 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_102 = torch._C._nn.gelu(x_101)
        x_101 = None
        x_103 = torch.nn.functional.dropout(x_102, 0.0, False, False)
        x_102 = None
        x_104 = torch.conv2d(
            x_103,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_103 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_105 = x_97 + x_104
        x_97 = x_104 = None
        x_106 = x_105.permute(0, 2, 3, 1)
        x_107 = torch.nn.functional.layer_norm(
            x_106,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_106 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_ = (None)
        x_108 = x_107.permute(0, 3, 1, 2)
        x_107 = None
        conv2d_42 = torch.conv2d(
            x_108,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_108 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_6 = conv2d_42.view(1, 12, 96, -1)
        conv2d_42 = None
        chunk_3 = view_6.chunk(3, dim=2)
        view_6 = None
        q_3 = chunk_3[0]
        k_3 = chunk_3[1]
        v_3 = chunk_3[2]
        chunk_3 = None
        relative_position_bias_6 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_7 = relative_position_bias_6.view((196, 196, 12))
        relative_position_bias_6 = None
        relative_position_bias_7 = view_7.permute(2, 0, 1)
        view_7 = None
        unsqueeze_3 = relative_position_bias_7.unsqueeze(0)
        relative_position_bias_7 = None
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
        x_109 = transpose_15.reshape(1, -1, 14, 14)
        transpose_15 = None
        x_110 = torch.conv2d(
            x_109,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_109 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_ = (None)
        x_111 = torch.nn.functional.dropout(x_110, 0.0, False, False)
        x_110 = None
        x_112 = x_105 + x_111
        x_105 = x_111 = None
        x_113 = x_112.permute(0, 2, 3, 1)
        x_114 = torch.nn.functional.layer_norm(
            x_113,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_113 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_ = (None)
        x_115 = x_114.permute(0, 3, 1, 2)
        x_114 = None
        x_116 = torch.conv2d(
            x_115,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_115 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_117 = torch._C._nn.gelu(x_116)
        x_116 = None
        x_118 = torch.nn.functional.dropout(x_117, 0.0, False, False)
        x_117 = None
        x_119 = torch.conv2d(
            x_118,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_118 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_120 = x_112 + x_119
        x_112 = x_119 = None
        x_121 = x_120.permute(0, 2, 3, 1)
        x_122 = torch.nn.functional.layer_norm(
            x_121,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_121 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_ = (None)
        x_123 = x_122.permute(0, 3, 1, 2)
        x_122 = None
        conv2d_46 = torch.conv2d(
            x_123,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_123 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_8 = conv2d_46.view(1, 12, 96, -1)
        conv2d_46 = None
        chunk_4 = view_8.chunk(3, dim=2)
        view_8 = None
        q_4 = chunk_4[0]
        k_4 = chunk_4[1]
        v_4 = chunk_4[2]
        chunk_4 = None
        relative_position_bias_8 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_9 = relative_position_bias_8.view((196, 196, 12))
        relative_position_bias_8 = None
        relative_position_bias_9 = view_9.permute(2, 0, 1)
        view_9 = None
        unsqueeze_4 = relative_position_bias_9.unsqueeze(0)
        relative_position_bias_9 = None
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
        x_124 = transpose_19.reshape(1, -1, 14, 14)
        transpose_19 = None
        x_125 = torch.conv2d(
            x_124,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_124 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_ = (None)
        x_126 = torch.nn.functional.dropout(x_125, 0.0, False, False)
        x_125 = None
        x_127 = x_120 + x_126
        x_120 = x_126 = None
        x_128 = x_127.permute(0, 2, 3, 1)
        x_129 = torch.nn.functional.layer_norm(
            x_128,
            (384,),
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_128 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_ = (None)
        x_130 = x_129.permute(0, 3, 1, 2)
        x_129 = None
        x_131 = torch.conv2d(
            x_130,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_130 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_132 = torch._C._nn.gelu(x_131)
        x_131 = None
        x_133 = torch.nn.functional.dropout(x_132, 0.0, False, False)
        x_132 = None
        x_134 = torch.conv2d(
            x_133,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_133 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_135 = x_127 + x_134
        x_127 = x_134 = None
        x_136 = torch._C._nn.avg_pool2d(x_135, 2, 2, 0, False, True, None)
        x_137 = torch.conv2d(
            x_136,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_136 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_ = (None)
        x_138 = x_135.permute(0, 2, 3, 1)
        x_135 = None
        x_139 = torch.nn.functional.layer_norm(
            x_138,
            (384,),
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_,
            1e-06,
        )
        x_138 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_ = (None)
        x_140 = x_139.permute(0, 3, 1, 2)
        x_139 = None
        x_141 = torch._C._nn.avg_pool2d(x_140, 2, 2, 0, False, True, None)
        x_140 = None
        conv2d_51 = torch.conv2d(
            x_141,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_141 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_10 = conv2d_51.view(1, 24, 96, -1)
        conv2d_51 = None
        chunk_5 = view_10.chunk(3, dim=2)
        view_10 = None
        q_5 = chunk_5[0]
        k_5 = chunk_5[1]
        v_5 = chunk_5[2]
        chunk_5 = None
        relative_position_bias_10 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_11 = relative_position_bias_10.view((49, 49, 24))
        relative_position_bias_10 = None
        relative_position_bias_11 = view_11.permute(2, 0, 1)
        view_11 = None
        unsqueeze_5 = relative_position_bias_11.unsqueeze(0)
        relative_position_bias_11 = None
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
        x_142 = transpose_23.reshape(1, -1, 7, 7)
        transpose_23 = None
        x_143 = torch.conv2d(
            x_142,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_142 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_144 = torch.nn.functional.dropout(x_143, 0.0, False, False)
        x_143 = None
        x_145 = x_137 + x_144
        x_137 = x_144 = None
        x_146 = x_145.permute(0, 2, 3, 1)
        x_147 = torch.nn.functional.layer_norm(
            x_146,
            (768,),
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_146 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_148 = x_147.permute(0, 3, 1, 2)
        x_147 = None
        x_149 = torch.conv2d(
            x_148,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_148 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_150 = torch._C._nn.gelu(x_149)
        x_149 = None
        x_151 = torch.nn.functional.dropout(x_150, 0.0, False, False)
        x_150 = None
        x_152 = torch.conv2d(
            x_151,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_151 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_153 = x_145 + x_152
        x_145 = x_152 = None
        x_154 = x_153.permute(0, 2, 3, 1)
        x_155 = torch.nn.functional.layer_norm(
            x_154,
            (768,),
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_154 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_156 = x_155.permute(0, 3, 1, 2)
        x_155 = None
        conv2d_55 = torch.conv2d(
            x_156,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_156 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_12 = conv2d_55.view(1, 24, 96, -1)
        conv2d_55 = None
        chunk_6 = view_12.chunk(3, dim=2)
        view_12 = None
        q_6 = chunk_6[0]
        k_6 = chunk_6[1]
        v_6 = chunk_6[2]
        chunk_6 = None
        relative_position_bias_12 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_13 = relative_position_bias_12.view((49, 49, 24))
        relative_position_bias_12 = None
        relative_position_bias_13 = view_13.permute(2, 0, 1)
        view_13 = None
        unsqueeze_6 = relative_position_bias_13.unsqueeze(0)
        relative_position_bias_13 = None
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
        x_157 = transpose_27.reshape(1, -1, 7, 7)
        transpose_27 = None
        x_158 = torch.conv2d(
            x_157,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_157 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_159 = torch.nn.functional.dropout(x_158, 0.0, False, False)
        x_158 = None
        x_160 = x_153 + x_159
        x_153 = x_159 = None
        x_161 = x_160.permute(0, 2, 3, 1)
        x_162 = torch.nn.functional.layer_norm(
            x_161,
            (768,),
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_161 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_163 = x_162.permute(0, 3, 1, 2)
        x_162 = None
        x_164 = torch.conv2d(
            x_163,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_163 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_165 = torch._C._nn.gelu(x_164)
        x_164 = None
        x_166 = torch.nn.functional.dropout(x_165, 0.0, False, False)
        x_165 = None
        x_167 = torch.conv2d(
            x_166,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_166 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_168 = x_160 + x_167
        x_160 = x_167 = None
        x_169 = torch.nn.functional.adaptive_avg_pool2d(x_168, 1)
        x_168 = None
        x_170 = x_169.permute(0, 2, 3, 1)
        x_169 = None
        x_171 = torch.nn.functional.layer_norm(
            x_170,
            (768,),
            l_self_modules_head_modules_norm_parameters_weight_,
            l_self_modules_head_modules_norm_parameters_bias_,
            1e-06,
        )
        x_170 = (
            l_self_modules_head_modules_norm_parameters_weight_
        ) = l_self_modules_head_modules_norm_parameters_bias_ = None
        x_172 = x_171.permute(0, 3, 1, 2)
        x_171 = None
        x_173 = x_172.flatten(1, -1)
        x_172 = None
        input_1 = torch._C._nn.linear(
            x_173,
            l_self_modules_head_modules_pre_logits_modules_fc_parameters_weight_,
            l_self_modules_head_modules_pre_logits_modules_fc_parameters_bias_,
        )
        x_173 = (
            l_self_modules_head_modules_pre_logits_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_pre_logits_modules_fc_parameters_bias_ = None
        input_2 = input_1.tanh()
        input_1 = None
        x_174 = torch.nn.functional.dropout(input_2, 0.0, False, False)
        input_2 = None
        x_175 = torch._C._nn.linear(
            x_174,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_174 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_175,)
