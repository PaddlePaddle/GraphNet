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
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_kxk_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_kxk_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_1x1_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_rel_pos_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_1x1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_1x1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_kxk_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_kxk_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_1x1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_1x1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_1x1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_1x1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_kxk_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_kxk_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_1x1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_1x1_parameters_weight_
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
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
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
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_rel_pos_buffers_relative_position_index_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_rel_pos_buffers_relative_position_index_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_
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
        x_6 = torch.nn.functional.silu(x_5, inplace=True)
        x_5 = None
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
        x_9 = torch.nn.functional.silu(x_8, inplace=True)
        x_8 = None
        x_10 = torch.conv2d(
            x_9,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_kxk_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            512,
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
        x_12 = torch.nn.functional.silu(x_11, inplace=True)
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
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_13 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_1x1_parameters_weight_ = (None)
        x_15 = x_14 + x_4
        x_14 = x_4 = None
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
        x_17 = torch.nn.functional.silu(x_16, inplace=True)
        x_16 = None
        x_18 = torch.conv2d(
            x_17,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_17 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_1x1_parameters_weight_ = (None)
        x_19 = torch.nn.functional.batch_norm(
            x_18,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_18 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_20 = torch.nn.functional.silu(x_19, inplace=True)
        x_19 = None
        x_21 = torch.conv2d(
            x_20,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        x_20 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_kxk_parameters_weight_ = (None)
        x_22 = torch.nn.functional.batch_norm(
            x_21,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_21 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_23 = torch.nn.functional.silu(x_22, inplace=True)
        x_22 = None
        x_se_4 = x_23.mean((2, 3), keepdim=True)
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
        x_24 = x_23 * sigmoid_1
        x_23 = sigmoid_1 = None
        x_25 = torch.conv2d(
            x_24,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_24 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_1x1_parameters_weight_ = (None)
        x_26 = x_25 + x_15
        x_25 = x_15 = None
        x_27 = torch._C._nn.avg_pool2d(x_26, 2, 2, 0, False, True, None)
        x_28 = torch.conv2d(
            x_27,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_27 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_ = (None)
        x_29 = torch.nn.functional.batch_norm(
            x_26,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_26 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_pre_norm_parameters_bias_ = (None)
        x_30 = torch.nn.functional.silu(x_29, inplace=True)
        x_29 = None
        x_31 = torch.conv2d(
            x_30,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_30 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_1x1_parameters_weight_ = (None)
        x_32 = torch.nn.functional.batch_norm(
            x_31,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_31 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (None)
        x_33 = torch.nn.functional.silu(x_32, inplace=True)
        x_32 = None
        x_34 = torch.conv2d(
            x_33,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_kxk_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            512,
        )
        x_33 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_kxk_parameters_weight_ = (None)
        x_35 = torch.nn.functional.batch_norm(
            x_34,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_34 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_36 = torch.nn.functional.silu(x_35, inplace=True)
        x_35 = None
        x_se_8 = x_36.mean((2, 3), keepdim=True)
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
        x_37 = x_36 * sigmoid_2
        x_36 = sigmoid_2 = None
        x_38 = torch.conv2d(
            x_37,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_37 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_1x1_parameters_weight_ = (None)
        x_39 = x_38 + x_28
        x_38 = x_28 = None
        x_40 = torch.nn.functional.batch_norm(
            x_39,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_pre_norm_parameters_bias_ = (None)
        x_41 = torch.nn.functional.silu(x_40, inplace=True)
        x_40 = None
        x_42 = torch.conv2d(
            x_41,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_41 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_1x1_parameters_weight_ = (None)
        x_43 = torch.nn.functional.batch_norm(
            x_42,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_42 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_44 = torch.nn.functional.silu(x_43, inplace=True)
        x_43 = None
        x_45 = torch.conv2d(
            x_44,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1024,
        )
        x_44 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_kxk_parameters_weight_ = (None)
        x_46 = torch.nn.functional.batch_norm(
            x_45,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_45 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_47 = torch.nn.functional.silu(x_46, inplace=True)
        x_46 = None
        x_se_12 = x_47.mean((2, 3), keepdim=True)
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
        x_48 = x_47 * sigmoid_3
        x_47 = sigmoid_3 = None
        x_49 = torch.conv2d(
            x_48,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_48 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_1x1_parameters_weight_ = (None)
        x_50 = x_49 + x_39
        x_49 = x_39 = None
        x_51 = torch.nn.functional.batch_norm(
            x_50,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_pre_norm_parameters_bias_ = (None)
        x_52 = torch.nn.functional.silu(x_51, inplace=True)
        x_51 = None
        x_53 = torch.conv2d(
            x_52,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_52 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_1x1_parameters_weight_ = (None)
        x_54 = torch.nn.functional.batch_norm(
            x_53,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_53 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_bias_ = (None)
        x_55 = torch.nn.functional.silu(x_54, inplace=True)
        x_54 = None
        x_56 = torch.conv2d(
            x_55,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1024,
        )
        x_55 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_kxk_parameters_weight_ = (None)
        x_57 = torch.nn.functional.batch_norm(
            x_56,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_56 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_bias_ = (None)
        x_58 = torch.nn.functional.silu(x_57, inplace=True)
        x_57 = None
        x_se_16 = x_58.mean((2, 3), keepdim=True)
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
        x_59 = x_58 * sigmoid_4
        x_58 = sigmoid_4 = None
        x_60 = torch.conv2d(
            x_59,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_59 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_1x1_parameters_weight_ = (None)
        x_61 = x_60 + x_50
        x_60 = x_50 = None
        x_62 = torch.nn.functional.batch_norm(
            x_61,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_pre_norm_parameters_bias_ = (None)
        x_63 = torch.nn.functional.silu(x_62, inplace=True)
        x_62 = None
        x_64 = torch.conv2d(
            x_63,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_63 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_1x1_parameters_weight_ = (None)
        x_65 = torch.nn.functional.batch_norm(
            x_64,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_64 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_bias_ = (None)
        x_66 = torch.nn.functional.silu(x_65, inplace=True)
        x_65 = None
        x_67 = torch.conv2d(
            x_66,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1024,
        )
        x_66 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_kxk_parameters_weight_ = (None)
        x_68 = torch.nn.functional.batch_norm(
            x_67,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_67 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_bias_ = (None)
        x_69 = torch.nn.functional.silu(x_68, inplace=True)
        x_68 = None
        x_se_20 = x_69.mean((2, 3), keepdim=True)
        x_se_21 = torch.conv2d(
            x_se_20,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_20 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc1_parameters_bias_ = (None)
        x_se_22 = torch.nn.functional.silu(x_se_21, inplace=True)
        x_se_21 = None
        x_se_23 = torch.conv2d(
            x_se_22,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_22 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_se_modules_fc2_parameters_bias_ = (None)
        sigmoid_5 = x_se_23.sigmoid()
        x_se_23 = None
        x_70 = x_69 * sigmoid_5
        x_69 = sigmoid_5 = None
        x_71 = torch.conv2d(
            x_70,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_70 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_1x1_parameters_weight_ = (None)
        x_72 = x_71 + x_61
        x_71 = x_61 = None
        x_73 = torch.nn.functional.batch_norm(
            x_72,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_pre_norm_parameters_bias_ = (None)
        x_74 = torch.nn.functional.silu(x_73, inplace=True)
        x_73 = None
        x_75 = torch.conv2d(
            x_74,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_74 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_1x1_parameters_weight_ = (None)
        x_76 = torch.nn.functional.batch_norm(
            x_75,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_75 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_bias_ = (None)
        x_77 = torch.nn.functional.silu(x_76, inplace=True)
        x_76 = None
        x_78 = torch.conv2d(
            x_77,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1024,
        )
        x_77 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_kxk_parameters_weight_ = (None)
        x_79 = torch.nn.functional.batch_norm(
            x_78,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_78 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_bias_ = (None)
        x_80 = torch.nn.functional.silu(x_79, inplace=True)
        x_79 = None
        x_se_24 = x_80.mean((2, 3), keepdim=True)
        x_se_25 = torch.conv2d(
            x_se_24,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_24 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc1_parameters_bias_ = (None)
        x_se_26 = torch.nn.functional.silu(x_se_25, inplace=True)
        x_se_25 = None
        x_se_27 = torch.conv2d(
            x_se_26,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_26 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_se_modules_fc2_parameters_bias_ = (None)
        sigmoid_6 = x_se_27.sigmoid()
        x_se_27 = None
        x_81 = x_80 * sigmoid_6
        x_80 = sigmoid_6 = None
        x_82 = torch.conv2d(
            x_81,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_81 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_1x1_parameters_weight_ = (None)
        x_83 = x_82 + x_72
        x_82 = x_72 = None
        x_84 = torch.nn.functional.batch_norm(
            x_83,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_pre_norm_parameters_bias_ = (None)
        x_85 = torch.nn.functional.silu(x_84, inplace=True)
        x_84 = None
        x_86 = torch.conv2d(
            x_85,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_85 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_1x1_parameters_weight_ = (None)
        x_87 = torch.nn.functional.batch_norm(
            x_86,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_86 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_bias_ = (None)
        x_88 = torch.nn.functional.silu(x_87, inplace=True)
        x_87 = None
        x_89 = torch.conv2d(
            x_88,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_kxk_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1024,
        )
        x_88 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_kxk_parameters_weight_ = (None)
        x_90 = torch.nn.functional.batch_norm(
            x_89,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_89 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_bias_ = (None)
        x_91 = torch.nn.functional.silu(x_90, inplace=True)
        x_90 = None
        x_se_28 = x_91.mean((2, 3), keepdim=True)
        x_se_29 = torch.conv2d(
            x_se_28,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_28 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc1_parameters_bias_ = (None)
        x_se_30 = torch.nn.functional.silu(x_se_29, inplace=True)
        x_se_29 = None
        x_se_31 = torch.conv2d(
            x_se_30,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_30 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_se_modules_fc2_parameters_bias_ = (None)
        sigmoid_7 = x_se_31.sigmoid()
        x_se_31 = None
        x_92 = x_91 * sigmoid_7
        x_91 = sigmoid_7 = None
        x_93 = torch.conv2d(
            x_92,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_1x1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_92 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_1x1_parameters_weight_ = (None)
        x_94 = x_93 + x_83
        x_93 = x_83 = None
        x_95 = torch._C._nn.avg_pool2d(x_94, 2, 2, 0, False, True, None)
        x_96 = torch.conv2d(
            x_95,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_95 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_ = (None)
        x_97 = x_94.permute(0, 2, 3, 1)
        x_94 = None
        x_98 = torch.nn.functional.layer_norm(
            x_97,
            (256,),
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_,
            1e-06,
        )
        x_97 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_ = (None)
        x_99 = x_98.permute(0, 3, 1, 2)
        x_98 = None
        x_100 = torch._C._nn.avg_pool2d(x_99, 2, 2, 0, False, True, None)
        x_99 = None
        conv2d_44 = torch.conv2d(
            x_100,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_100 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        view = conv2d_44.view(1, 8, 96, -1)
        conv2d_44 = None
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
        view_1 = relative_position_bias.view((196, 196, 8))
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
        x_101 = transpose_3.reshape(1, -1, 14, 14)
        transpose_3 = None
        x_102 = torch.conv2d(
            x_101,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_101 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_103 = torch.nn.functional.dropout(x_102, 0.0, False, False)
        x_102 = None
        x_104 = x_96 + x_103
        x_96 = x_103 = None
        x_105 = x_104.permute(0, 2, 3, 1)
        x_106 = torch.nn.functional.layer_norm(
            x_105,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_105 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_107 = x_106.permute(0, 3, 1, 2)
        x_106 = None
        x_108 = torch.conv2d(
            x_107,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_107 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_109 = torch._C._nn.gelu(x_108)
        x_108 = None
        x_110 = torch.nn.functional.dropout(x_109, 0.0, False, False)
        x_109 = None
        x_111 = torch.conv2d(
            x_110,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_110 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_112 = x_104 + x_111
        x_104 = x_111 = None
        x_113 = x_112.permute(0, 2, 3, 1)
        x_114 = torch.nn.functional.layer_norm(
            x_113,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_113 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_115 = x_114.permute(0, 3, 1, 2)
        x_114 = None
        conv2d_48 = torch.conv2d(
            x_115,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_115 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_2 = conv2d_48.view(1, 16, 96, -1)
        conv2d_48 = None
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
        view_3 = relative_position_bias_2.view((196, 196, 16))
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
        x_116 = transpose_7.reshape(1, -1, 14, 14)
        transpose_7 = None
        x_117 = torch.conv2d(
            x_116,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_116 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_118 = torch.nn.functional.dropout(x_117, 0.0, False, False)
        x_117 = None
        x_119 = x_112 + x_118
        x_112 = x_118 = None
        x_120 = x_119.permute(0, 2, 3, 1)
        x_121 = torch.nn.functional.layer_norm(
            x_120,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_120 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_122 = x_121.permute(0, 3, 1, 2)
        x_121 = None
        x_123 = torch.conv2d(
            x_122,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_122 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_124 = torch._C._nn.gelu(x_123)
        x_123 = None
        x_125 = torch.nn.functional.dropout(x_124, 0.0, False, False)
        x_124 = None
        x_126 = torch.conv2d(
            x_125,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_125 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_127 = x_119 + x_126
        x_119 = x_126 = None
        x_128 = x_127.permute(0, 2, 3, 1)
        x_129 = torch.nn.functional.layer_norm(
            x_128,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_128 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_ = (None)
        x_130 = x_129.permute(0, 3, 1, 2)
        x_129 = None
        conv2d_52 = torch.conv2d(
            x_130,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_130 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_4 = conv2d_52.view(1, 16, 96, -1)
        conv2d_52 = None
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
        view_5 = relative_position_bias_4.view((196, 196, 16))
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
        x_131 = transpose_11.reshape(1, -1, 14, 14)
        transpose_11 = None
        x_132 = torch.conv2d(
            x_131,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_131 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_ = (None)
        x_133 = torch.nn.functional.dropout(x_132, 0.0, False, False)
        x_132 = None
        x_134 = x_127 + x_133
        x_127 = x_133 = None
        x_135 = x_134.permute(0, 2, 3, 1)
        x_136 = torch.nn.functional.layer_norm(
            x_135,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_135 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_ = (None)
        x_137 = x_136.permute(0, 3, 1, 2)
        x_136 = None
        x_138 = torch.conv2d(
            x_137,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_137 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_139 = torch._C._nn.gelu(x_138)
        x_138 = None
        x_140 = torch.nn.functional.dropout(x_139, 0.0, False, False)
        x_139 = None
        x_141 = torch.conv2d(
            x_140,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_140 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_142 = x_134 + x_141
        x_134 = x_141 = None
        x_143 = x_142.permute(0, 2, 3, 1)
        x_144 = torch.nn.functional.layer_norm(
            x_143,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_143 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_ = (None)
        x_145 = x_144.permute(0, 3, 1, 2)
        x_144 = None
        conv2d_56 = torch.conv2d(
            x_145,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_145 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_6 = conv2d_56.view(1, 16, 96, -1)
        conv2d_56 = None
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
        view_7 = relative_position_bias_6.view((196, 196, 16))
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
        x_146 = transpose_15.reshape(1, -1, 14, 14)
        transpose_15 = None
        x_147 = torch.conv2d(
            x_146,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_146 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_ = (None)
        x_148 = torch.nn.functional.dropout(x_147, 0.0, False, False)
        x_147 = None
        x_149 = x_142 + x_148
        x_142 = x_148 = None
        x_150 = x_149.permute(0, 2, 3, 1)
        x_151 = torch.nn.functional.layer_norm(
            x_150,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_150 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_ = (None)
        x_152 = x_151.permute(0, 3, 1, 2)
        x_151 = None
        x_153 = torch.conv2d(
            x_152,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_152 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_154 = torch._C._nn.gelu(x_153)
        x_153 = None
        x_155 = torch.nn.functional.dropout(x_154, 0.0, False, False)
        x_154 = None
        x_156 = torch.conv2d(
            x_155,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_155 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_157 = x_149 + x_156
        x_149 = x_156 = None
        x_158 = x_157.permute(0, 2, 3, 1)
        x_159 = torch.nn.functional.layer_norm(
            x_158,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_158 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_ = (None)
        x_160 = x_159.permute(0, 3, 1, 2)
        x_159 = None
        conv2d_60 = torch.conv2d(
            x_160,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_160 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_8 = conv2d_60.view(1, 16, 96, -1)
        conv2d_60 = None
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
        view_9 = relative_position_bias_8.view((196, 196, 16))
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
        x_161 = transpose_19.reshape(1, -1, 14, 14)
        transpose_19 = None
        x_162 = torch.conv2d(
            x_161,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_161 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_ = (None)
        x_163 = torch.nn.functional.dropout(x_162, 0.0, False, False)
        x_162 = None
        x_164 = x_157 + x_163
        x_157 = x_163 = None
        x_165 = x_164.permute(0, 2, 3, 1)
        x_166 = torch.nn.functional.layer_norm(
            x_165,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_165 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_ = (None)
        x_167 = x_166.permute(0, 3, 1, 2)
        x_166 = None
        x_168 = torch.conv2d(
            x_167,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_167 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_169 = torch._C._nn.gelu(x_168)
        x_168 = None
        x_170 = torch.nn.functional.dropout(x_169, 0.0, False, False)
        x_169 = None
        x_171 = torch.conv2d(
            x_170,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_170 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_172 = x_164 + x_171
        x_164 = x_171 = None
        x_173 = x_172.permute(0, 2, 3, 1)
        x_174 = torch.nn.functional.layer_norm(
            x_173,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_173 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_ = (None)
        x_175 = x_174.permute(0, 3, 1, 2)
        x_174 = None
        conv2d_64 = torch.conv2d(
            x_175,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_175 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_10 = conv2d_64.view(1, 16, 96, -1)
        conv2d_64 = None
        chunk_5 = view_10.chunk(3, dim=2)
        view_10 = None
        q_5 = chunk_5[0]
        k_5 = chunk_5[1]
        v_5 = chunk_5[2]
        chunk_5 = None
        relative_position_bias_10 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_11 = relative_position_bias_10.view((196, 196, 16))
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
        x_176 = transpose_23.reshape(1, -1, 14, 14)
        transpose_23 = None
        x_177 = torch.conv2d(
            x_176,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_176 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_ = (None)
        x_178 = torch.nn.functional.dropout(x_177, 0.0, False, False)
        x_177 = None
        x_179 = x_172 + x_178
        x_172 = x_178 = None
        x_180 = x_179.permute(0, 2, 3, 1)
        x_181 = torch.nn.functional.layer_norm(
            x_180,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_180 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_ = (None)
        x_182 = x_181.permute(0, 3, 1, 2)
        x_181 = None
        x_183 = torch.conv2d(
            x_182,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_182 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_184 = torch._C._nn.gelu(x_183)
        x_183 = None
        x_185 = torch.nn.functional.dropout(x_184, 0.0, False, False)
        x_184 = None
        x_186 = torch.conv2d(
            x_185,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_185 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_187 = x_179 + x_186
        x_179 = x_186 = None
        x_188 = x_187.permute(0, 2, 3, 1)
        x_189 = torch.nn.functional.layer_norm(
            x_188,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_188 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_bias_ = (None)
        x_190 = x_189.permute(0, 3, 1, 2)
        x_189 = None
        conv2d_68 = torch.conv2d(
            x_190,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_190 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_12 = conv2d_68.view(1, 16, 96, -1)
        conv2d_68 = None
        chunk_6 = view_12.chunk(3, dim=2)
        view_12 = None
        q_6 = chunk_6[0]
        k_6 = chunk_6[1]
        v_6 = chunk_6[2]
        chunk_6 = None
        relative_position_bias_12 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_13 = relative_position_bias_12.view((196, 196, 16))
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
        x_191 = transpose_27.reshape(1, -1, 14, 14)
        transpose_27 = None
        x_192 = torch.conv2d(
            x_191,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_191 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_ = (None)
        x_193 = torch.nn.functional.dropout(x_192, 0.0, False, False)
        x_192 = None
        x_194 = x_187 + x_193
        x_187 = x_193 = None
        x_195 = x_194.permute(0, 2, 3, 1)
        x_196 = torch.nn.functional.layer_norm(
            x_195,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_195 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_bias_ = (None)
        x_197 = x_196.permute(0, 3, 1, 2)
        x_196 = None
        x_198 = torch.conv2d(
            x_197,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_197 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_199 = torch._C._nn.gelu(x_198)
        x_198 = None
        x_200 = torch.nn.functional.dropout(x_199, 0.0, False, False)
        x_199 = None
        x_201 = torch.conv2d(
            x_200,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_200 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_202 = x_194 + x_201
        x_194 = x_201 = None
        x_203 = x_202.permute(0, 2, 3, 1)
        x_204 = torch.nn.functional.layer_norm(
            x_203,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_203 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_bias_ = (None)
        x_205 = x_204.permute(0, 3, 1, 2)
        x_204 = None
        conv2d_72 = torch.conv2d(
            x_205,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_205 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_14 = conv2d_72.view(1, 16, 96, -1)
        conv2d_72 = None
        chunk_7 = view_14.chunk(3, dim=2)
        view_14 = None
        q_7 = chunk_7[0]
        k_7 = chunk_7[1]
        v_7 = chunk_7[2]
        chunk_7 = None
        relative_position_bias_14 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_15 = relative_position_bias_14.view((196, 196, 16))
        relative_position_bias_14 = None
        relative_position_bias_15 = view_15.permute(2, 0, 1)
        view_15 = None
        unsqueeze_7 = relative_position_bias_15.unsqueeze(0)
        relative_position_bias_15 = None
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
        x_206 = transpose_31.reshape(1, -1, 14, 14)
        transpose_31 = None
        x_207 = torch.conv2d(
            x_206,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_206 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_ = (None)
        x_208 = torch.nn.functional.dropout(x_207, 0.0, False, False)
        x_207 = None
        x_209 = x_202 + x_208
        x_202 = x_208 = None
        x_210 = x_209.permute(0, 2, 3, 1)
        x_211 = torch.nn.functional.layer_norm(
            x_210,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_210 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_bias_ = (None)
        x_212 = x_211.permute(0, 3, 1, 2)
        x_211 = None
        x_213 = torch.conv2d(
            x_212,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_212 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_214 = torch._C._nn.gelu(x_213)
        x_213 = None
        x_215 = torch.nn.functional.dropout(x_214, 0.0, False, False)
        x_214 = None
        x_216 = torch.conv2d(
            x_215,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_215 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_217 = x_209 + x_216
        x_209 = x_216 = None
        x_218 = x_217.permute(0, 2, 3, 1)
        x_219 = torch.nn.functional.layer_norm(
            x_218,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_218 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_bias_ = (None)
        x_220 = x_219.permute(0, 3, 1, 2)
        x_219 = None
        conv2d_76 = torch.conv2d(
            x_220,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_220 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_16 = conv2d_76.view(1, 16, 96, -1)
        conv2d_76 = None
        chunk_8 = view_16.chunk(3, dim=2)
        view_16 = None
        q_8 = chunk_8[0]
        k_8 = chunk_8[1]
        v_8 = chunk_8[2]
        chunk_8 = None
        relative_position_bias_16 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_17 = relative_position_bias_16.view((196, 196, 16))
        relative_position_bias_16 = None
        relative_position_bias_17 = view_17.permute(2, 0, 1)
        view_17 = None
        unsqueeze_8 = relative_position_bias_17.unsqueeze(0)
        relative_position_bias_17 = None
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
        x_221 = transpose_35.reshape(1, -1, 14, 14)
        transpose_35 = None
        x_222 = torch.conv2d(
            x_221,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_221 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_ = (None)
        x_223 = torch.nn.functional.dropout(x_222, 0.0, False, False)
        x_222 = None
        x_224 = x_217 + x_223
        x_217 = x_223 = None
        x_225 = x_224.permute(0, 2, 3, 1)
        x_226 = torch.nn.functional.layer_norm(
            x_225,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_225 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_bias_ = (None)
        x_227 = x_226.permute(0, 3, 1, 2)
        x_226 = None
        x_228 = torch.conv2d(
            x_227,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_227 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_229 = torch._C._nn.gelu(x_228)
        x_228 = None
        x_230 = torch.nn.functional.dropout(x_229, 0.0, False, False)
        x_229 = None
        x_231 = torch.conv2d(
            x_230,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_230 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_232 = x_224 + x_231
        x_224 = x_231 = None
        x_233 = x_232.permute(0, 2, 3, 1)
        x_234 = torch.nn.functional.layer_norm(
            x_233,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_233 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_bias_ = (None)
        x_235 = x_234.permute(0, 3, 1, 2)
        x_234 = None
        conv2d_80 = torch.conv2d(
            x_235,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_235 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_18 = conv2d_80.view(1, 16, 96, -1)
        conv2d_80 = None
        chunk_9 = view_18.chunk(3, dim=2)
        view_18 = None
        q_9 = chunk_9[0]
        k_9 = chunk_9[1]
        v_9 = chunk_9[2]
        chunk_9 = None
        relative_position_bias_18 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_19 = relative_position_bias_18.view((196, 196, 16))
        relative_position_bias_18 = None
        relative_position_bias_19 = view_19.permute(2, 0, 1)
        view_19 = None
        unsqueeze_9 = relative_position_bias_19.unsqueeze(0)
        relative_position_bias_19 = None
        attn_bias_9 = unsqueeze_9.contiguous()
        unsqueeze_9 = None
        transpose_36 = q_9.transpose(-1, -2)
        q_9 = None
        contiguous_37 = transpose_36.contiguous()
        transpose_36 = None
        transpose_37 = k_9.transpose(-1, -2)
        k_9 = None
        contiguous_38 = transpose_37.contiguous()
        transpose_37 = None
        transpose_38 = v_9.transpose(-1, -2)
        v_9 = None
        contiguous_39 = transpose_38.contiguous()
        transpose_38 = None
        scaled_dot_product_attention_9 = torch._C._nn.scaled_dot_product_attention(
            contiguous_37,
            contiguous_38,
            contiguous_39,
            attn_mask=attn_bias_9,
            dropout_p=0.0,
        )
        contiguous_37 = contiguous_38 = contiguous_39 = attn_bias_9 = None
        transpose_39 = scaled_dot_product_attention_9.transpose(-1, -2)
        scaled_dot_product_attention_9 = None
        x_236 = transpose_39.reshape(1, -1, 14, 14)
        transpose_39 = None
        x_237 = torch.conv2d(
            x_236,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_236 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_ = (None)
        x_238 = torch.nn.functional.dropout(x_237, 0.0, False, False)
        x_237 = None
        x_239 = x_232 + x_238
        x_232 = x_238 = None
        x_240 = x_239.permute(0, 2, 3, 1)
        x_241 = torch.nn.functional.layer_norm(
            x_240,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_240 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_bias_ = (None)
        x_242 = x_241.permute(0, 3, 1, 2)
        x_241 = None
        x_243 = torch.conv2d(
            x_242,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_242 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_244 = torch._C._nn.gelu(x_243)
        x_243 = None
        x_245 = torch.nn.functional.dropout(x_244, 0.0, False, False)
        x_244 = None
        x_246 = torch.conv2d(
            x_245,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_245 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_247 = x_239 + x_246
        x_239 = x_246 = None
        x_248 = x_247.permute(0, 2, 3, 1)
        x_249 = torch.nn.functional.layer_norm(
            x_248,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_248 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_bias_ = (None)
        x_250 = x_249.permute(0, 3, 1, 2)
        x_249 = None
        conv2d_84 = torch.conv2d(
            x_250,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_250 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_20 = conv2d_84.view(1, 16, 96, -1)
        conv2d_84 = None
        chunk_10 = view_20.chunk(3, dim=2)
        view_20 = None
        q_10 = chunk_10[0]
        k_10 = chunk_10[1]
        v_10 = chunk_10[2]
        chunk_10 = None
        relative_position_bias_20 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_21 = relative_position_bias_20.view((196, 196, 16))
        relative_position_bias_20 = None
        relative_position_bias_21 = view_21.permute(2, 0, 1)
        view_21 = None
        unsqueeze_10 = relative_position_bias_21.unsqueeze(0)
        relative_position_bias_21 = None
        attn_bias_10 = unsqueeze_10.contiguous()
        unsqueeze_10 = None
        transpose_40 = q_10.transpose(-1, -2)
        q_10 = None
        contiguous_41 = transpose_40.contiguous()
        transpose_40 = None
        transpose_41 = k_10.transpose(-1, -2)
        k_10 = None
        contiguous_42 = transpose_41.contiguous()
        transpose_41 = None
        transpose_42 = v_10.transpose(-1, -2)
        v_10 = None
        contiguous_43 = transpose_42.contiguous()
        transpose_42 = None
        scaled_dot_product_attention_10 = torch._C._nn.scaled_dot_product_attention(
            contiguous_41,
            contiguous_42,
            contiguous_43,
            attn_mask=attn_bias_10,
            dropout_p=0.0,
        )
        contiguous_41 = contiguous_42 = contiguous_43 = attn_bias_10 = None
        transpose_43 = scaled_dot_product_attention_10.transpose(-1, -2)
        scaled_dot_product_attention_10 = None
        x_251 = transpose_43.reshape(1, -1, 14, 14)
        transpose_43 = None
        x_252 = torch.conv2d(
            x_251,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_251 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_ = (None)
        x_253 = torch.nn.functional.dropout(x_252, 0.0, False, False)
        x_252 = None
        x_254 = x_247 + x_253
        x_247 = x_253 = None
        x_255 = x_254.permute(0, 2, 3, 1)
        x_256 = torch.nn.functional.layer_norm(
            x_255,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_255 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_bias_ = (None)
        x_257 = x_256.permute(0, 3, 1, 2)
        x_256 = None
        x_258 = torch.conv2d(
            x_257,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_257 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_259 = torch._C._nn.gelu(x_258)
        x_258 = None
        x_260 = torch.nn.functional.dropout(x_259, 0.0, False, False)
        x_259 = None
        x_261 = torch.conv2d(
            x_260,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_260 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_262 = x_254 + x_261
        x_254 = x_261 = None
        x_263 = x_262.permute(0, 2, 3, 1)
        x_264 = torch.nn.functional.layer_norm(
            x_263,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_263 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_bias_ = (None)
        x_265 = x_264.permute(0, 3, 1, 2)
        x_264 = None
        conv2d_88 = torch.conv2d(
            x_265,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_265 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_22 = conv2d_88.view(1, 16, 96, -1)
        conv2d_88 = None
        chunk_11 = view_22.chunk(3, dim=2)
        view_22 = None
        q_11 = chunk_11[0]
        k_11 = chunk_11[1]
        v_11 = chunk_11[2]
        chunk_11 = None
        relative_position_bias_22 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_23 = relative_position_bias_22.view((196, 196, 16))
        relative_position_bias_22 = None
        relative_position_bias_23 = view_23.permute(2, 0, 1)
        view_23 = None
        unsqueeze_11 = relative_position_bias_23.unsqueeze(0)
        relative_position_bias_23 = None
        attn_bias_11 = unsqueeze_11.contiguous()
        unsqueeze_11 = None
        transpose_44 = q_11.transpose(-1, -2)
        q_11 = None
        contiguous_45 = transpose_44.contiguous()
        transpose_44 = None
        transpose_45 = k_11.transpose(-1, -2)
        k_11 = None
        contiguous_46 = transpose_45.contiguous()
        transpose_45 = None
        transpose_46 = v_11.transpose(-1, -2)
        v_11 = None
        contiguous_47 = transpose_46.contiguous()
        transpose_46 = None
        scaled_dot_product_attention_11 = torch._C._nn.scaled_dot_product_attention(
            contiguous_45,
            contiguous_46,
            contiguous_47,
            attn_mask=attn_bias_11,
            dropout_p=0.0,
        )
        contiguous_45 = contiguous_46 = contiguous_47 = attn_bias_11 = None
        transpose_47 = scaled_dot_product_attention_11.transpose(-1, -2)
        scaled_dot_product_attention_11 = None
        x_266 = transpose_47.reshape(1, -1, 14, 14)
        transpose_47 = None
        x_267 = torch.conv2d(
            x_266,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_266 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_ = (None)
        x_268 = torch.nn.functional.dropout(x_267, 0.0, False, False)
        x_267 = None
        x_269 = x_262 + x_268
        x_262 = x_268 = None
        x_270 = x_269.permute(0, 2, 3, 1)
        x_271 = torch.nn.functional.layer_norm(
            x_270,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_270 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_bias_ = (None)
        x_272 = x_271.permute(0, 3, 1, 2)
        x_271 = None
        x_273 = torch.conv2d(
            x_272,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_272 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_274 = torch._C._nn.gelu(x_273)
        x_273 = None
        x_275 = torch.nn.functional.dropout(x_274, 0.0, False, False)
        x_274 = None
        x_276 = torch.conv2d(
            x_275,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_275 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_277 = x_269 + x_276
        x_269 = x_276 = None
        x_278 = x_277.permute(0, 2, 3, 1)
        x_279 = torch.nn.functional.layer_norm(
            x_278,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_278 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_bias_ = (None)
        x_280 = x_279.permute(0, 3, 1, 2)
        x_279 = None
        conv2d_92 = torch.conv2d(
            x_280,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_280 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_24 = conv2d_92.view(1, 16, 96, -1)
        conv2d_92 = None
        chunk_12 = view_24.chunk(3, dim=2)
        view_24 = None
        q_12 = chunk_12[0]
        k_12 = chunk_12[1]
        v_12 = chunk_12[2]
        chunk_12 = None
        relative_position_bias_24 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_25 = relative_position_bias_24.view((196, 196, 16))
        relative_position_bias_24 = None
        relative_position_bias_25 = view_25.permute(2, 0, 1)
        view_25 = None
        unsqueeze_12 = relative_position_bias_25.unsqueeze(0)
        relative_position_bias_25 = None
        attn_bias_12 = unsqueeze_12.contiguous()
        unsqueeze_12 = None
        transpose_48 = q_12.transpose(-1, -2)
        q_12 = None
        contiguous_49 = transpose_48.contiguous()
        transpose_48 = None
        transpose_49 = k_12.transpose(-1, -2)
        k_12 = None
        contiguous_50 = transpose_49.contiguous()
        transpose_49 = None
        transpose_50 = v_12.transpose(-1, -2)
        v_12 = None
        contiguous_51 = transpose_50.contiguous()
        transpose_50 = None
        scaled_dot_product_attention_12 = torch._C._nn.scaled_dot_product_attention(
            contiguous_49,
            contiguous_50,
            contiguous_51,
            attn_mask=attn_bias_12,
            dropout_p=0.0,
        )
        contiguous_49 = contiguous_50 = contiguous_51 = attn_bias_12 = None
        transpose_51 = scaled_dot_product_attention_12.transpose(-1, -2)
        scaled_dot_product_attention_12 = None
        x_281 = transpose_51.reshape(1, -1, 14, 14)
        transpose_51 = None
        x_282 = torch.conv2d(
            x_281,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_281 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_ = (None)
        x_283 = torch.nn.functional.dropout(x_282, 0.0, False, False)
        x_282 = None
        x_284 = x_277 + x_283
        x_277 = x_283 = None
        x_285 = x_284.permute(0, 2, 3, 1)
        x_286 = torch.nn.functional.layer_norm(
            x_285,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_285 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_bias_ = (None)
        x_287 = x_286.permute(0, 3, 1, 2)
        x_286 = None
        x_288 = torch.conv2d(
            x_287,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_287 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_289 = torch._C._nn.gelu(x_288)
        x_288 = None
        x_290 = torch.nn.functional.dropout(x_289, 0.0, False, False)
        x_289 = None
        x_291 = torch.conv2d(
            x_290,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_290 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_292 = x_284 + x_291
        x_284 = x_291 = None
        x_293 = x_292.permute(0, 2, 3, 1)
        x_294 = torch.nn.functional.layer_norm(
            x_293,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_293 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_bias_ = (None)
        x_295 = x_294.permute(0, 3, 1, 2)
        x_294 = None
        conv2d_96 = torch.conv2d(
            x_295,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_295 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_26 = conv2d_96.view(1, 16, 96, -1)
        conv2d_96 = None
        chunk_13 = view_26.chunk(3, dim=2)
        view_26 = None
        q_13 = chunk_13[0]
        k_13 = chunk_13[1]
        v_13 = chunk_13[2]
        chunk_13 = None
        relative_position_bias_26 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_27 = relative_position_bias_26.view((196, 196, 16))
        relative_position_bias_26 = None
        relative_position_bias_27 = view_27.permute(2, 0, 1)
        view_27 = None
        unsqueeze_13 = relative_position_bias_27.unsqueeze(0)
        relative_position_bias_27 = None
        attn_bias_13 = unsqueeze_13.contiguous()
        unsqueeze_13 = None
        transpose_52 = q_13.transpose(-1, -2)
        q_13 = None
        contiguous_53 = transpose_52.contiguous()
        transpose_52 = None
        transpose_53 = k_13.transpose(-1, -2)
        k_13 = None
        contiguous_54 = transpose_53.contiguous()
        transpose_53 = None
        transpose_54 = v_13.transpose(-1, -2)
        v_13 = None
        contiguous_55 = transpose_54.contiguous()
        transpose_54 = None
        scaled_dot_product_attention_13 = torch._C._nn.scaled_dot_product_attention(
            contiguous_53,
            contiguous_54,
            contiguous_55,
            attn_mask=attn_bias_13,
            dropout_p=0.0,
        )
        contiguous_53 = contiguous_54 = contiguous_55 = attn_bias_13 = None
        transpose_55 = scaled_dot_product_attention_13.transpose(-1, -2)
        scaled_dot_product_attention_13 = None
        x_296 = transpose_55.reshape(1, -1, 14, 14)
        transpose_55 = None
        x_297 = torch.conv2d(
            x_296,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_296 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_ = (None)
        x_298 = torch.nn.functional.dropout(x_297, 0.0, False, False)
        x_297 = None
        x_299 = x_292 + x_298
        x_292 = x_298 = None
        x_300 = x_299.permute(0, 2, 3, 1)
        x_301 = torch.nn.functional.layer_norm(
            x_300,
            (512,),
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_300 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_bias_ = (None)
        x_302 = x_301.permute(0, 3, 1, 2)
        x_301 = None
        x_303 = torch.conv2d(
            x_302,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_302 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_304 = torch._C._nn.gelu(x_303)
        x_303 = None
        x_305 = torch.nn.functional.dropout(x_304, 0.0, False, False)
        x_304 = None
        x_306 = torch.conv2d(
            x_305,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_305 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_307 = x_299 + x_306
        x_299 = x_306 = None
        x_308 = torch._C._nn.avg_pool2d(x_307, 2, 2, 0, False, True, None)
        x_309 = torch.conv2d(
            x_308,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_308 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_ = (None)
        x_310 = x_307.permute(0, 2, 3, 1)
        x_307 = None
        x_311 = torch.nn.functional.layer_norm(
            x_310,
            (512,),
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_,
            1e-06,
        )
        x_310 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_ = (None)
        x_312 = x_311.permute(0, 3, 1, 2)
        x_311 = None
        x_313 = torch._C._nn.avg_pool2d(x_312, 2, 2, 0, False, True, None)
        x_312 = None
        conv2d_101 = torch.conv2d(
            x_313,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_313 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_28 = conv2d_101.view(1, 16, 96, -1)
        conv2d_101 = None
        chunk_14 = view_28.chunk(3, dim=2)
        view_28 = None
        q_14 = chunk_14[0]
        k_14 = chunk_14[1]
        v_14 = chunk_14[2]
        chunk_14 = None
        relative_position_bias_28 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_29 = relative_position_bias_28.view((49, 49, 16))
        relative_position_bias_28 = None
        relative_position_bias_29 = view_29.permute(2, 0, 1)
        view_29 = None
        unsqueeze_14 = relative_position_bias_29.unsqueeze(0)
        relative_position_bias_29 = None
        attn_bias_14 = unsqueeze_14.contiguous()
        unsqueeze_14 = None
        transpose_56 = q_14.transpose(-1, -2)
        q_14 = None
        contiguous_57 = transpose_56.contiguous()
        transpose_56 = None
        transpose_57 = k_14.transpose(-1, -2)
        k_14 = None
        contiguous_58 = transpose_57.contiguous()
        transpose_57 = None
        transpose_58 = v_14.transpose(-1, -2)
        v_14 = None
        contiguous_59 = transpose_58.contiguous()
        transpose_58 = None
        scaled_dot_product_attention_14 = torch._C._nn.scaled_dot_product_attention(
            contiguous_57,
            contiguous_58,
            contiguous_59,
            attn_mask=attn_bias_14,
            dropout_p=0.0,
        )
        contiguous_57 = contiguous_58 = contiguous_59 = attn_bias_14 = None
        transpose_59 = scaled_dot_product_attention_14.transpose(-1, -2)
        scaled_dot_product_attention_14 = None
        x_314 = transpose_59.reshape(1, -1, 7, 7)
        transpose_59 = None
        x_315 = torch.conv2d(
            x_314,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_314 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_316 = torch.nn.functional.dropout(x_315, 0.0, False, False)
        x_315 = None
        x_317 = x_309 + x_316
        x_309 = x_316 = None
        x_318 = x_317.permute(0, 2, 3, 1)
        x_319 = torch.nn.functional.layer_norm(
            x_318,
            (1024,),
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_318 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_320 = x_319.permute(0, 3, 1, 2)
        x_319 = None
        x_321 = torch.conv2d(
            x_320,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_320 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_322 = torch._C._nn.gelu(x_321)
        x_321 = None
        x_323 = torch.nn.functional.dropout(x_322, 0.0, False, False)
        x_322 = None
        x_324 = torch.conv2d(
            x_323,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_323 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_325 = x_317 + x_324
        x_317 = x_324 = None
        x_326 = x_325.permute(0, 2, 3, 1)
        x_327 = torch.nn.functional.layer_norm(
            x_326,
            (1024,),
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_326 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_328 = x_327.permute(0, 3, 1, 2)
        x_327 = None
        conv2d_105 = torch.conv2d(
            x_328,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_328 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        view_30 = conv2d_105.view(1, 32, 96, -1)
        conv2d_105 = None
        chunk_15 = view_30.chunk(3, dim=2)
        view_30 = None
        q_15 = chunk_15[0]
        k_15 = chunk_15[1]
        v_15 = chunk_15[2]
        chunk_15 = None
        relative_position_bias_30 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_[
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_relative_position_index_
        ]
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_relative_position_index_ = (None)
        view_31 = relative_position_bias_30.view((49, 49, 32))
        relative_position_bias_30 = None
        relative_position_bias_31 = view_31.permute(2, 0, 1)
        view_31 = None
        unsqueeze_15 = relative_position_bias_31.unsqueeze(0)
        relative_position_bias_31 = None
        attn_bias_15 = unsqueeze_15.contiguous()
        unsqueeze_15 = None
        transpose_60 = q_15.transpose(-1, -2)
        q_15 = None
        contiguous_61 = transpose_60.contiguous()
        transpose_60 = None
        transpose_61 = k_15.transpose(-1, -2)
        k_15 = None
        contiguous_62 = transpose_61.contiguous()
        transpose_61 = None
        transpose_62 = v_15.transpose(-1, -2)
        v_15 = None
        contiguous_63 = transpose_62.contiguous()
        transpose_62 = None
        scaled_dot_product_attention_15 = torch._C._nn.scaled_dot_product_attention(
            contiguous_61,
            contiguous_62,
            contiguous_63,
            attn_mask=attn_bias_15,
            dropout_p=0.0,
        )
        contiguous_61 = contiguous_62 = contiguous_63 = attn_bias_15 = None
        transpose_63 = scaled_dot_product_attention_15.transpose(-1, -2)
        scaled_dot_product_attention_15 = None
        x_329 = transpose_63.reshape(1, -1, 7, 7)
        transpose_63 = None
        x_330 = torch.conv2d(
            x_329,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_329 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_331 = torch.nn.functional.dropout(x_330, 0.0, False, False)
        x_330 = None
        x_332 = x_325 + x_331
        x_325 = x_331 = None
        x_333 = x_332.permute(0, 2, 3, 1)
        x_334 = torch.nn.functional.layer_norm(
            x_333,
            (1024,),
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_333 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_335 = x_334.permute(0, 3, 1, 2)
        x_334 = None
        x_336 = torch.conv2d(
            x_335,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_335 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_337 = torch._C._nn.gelu(x_336)
        x_336 = None
        x_338 = torch.nn.functional.dropout(x_337, 0.0, False, False)
        x_337 = None
        x_339 = torch.conv2d(
            x_338,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_338 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_340 = x_332 + x_339
        x_332 = x_339 = None
        x_341 = x_340.permute(0, 2, 3, 1)
        x_340 = None
        x_342 = torch.nn.functional.layer_norm(
            x_341,
            (1024,),
            l_self_modules_norm_parameters_weight_,
            l_self_modules_norm_parameters_bias_,
            1e-06,
        )
        x_341 = (
            l_self_modules_norm_parameters_weight_
        ) = l_self_modules_norm_parameters_bias_ = None
        x_343 = x_342.permute(0, 3, 1, 2)
        x_342 = None
        x_344 = torch.nn.functional.adaptive_avg_pool2d(x_343, 1)
        x_343 = None
        x_345 = x_344.flatten(1, -1)
        x_344 = None
        x_346 = torch.nn.functional.dropout(x_345, 0.0, False, False)
        x_345 = None
        x_347 = torch._C._nn.linear(
            x_346,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_346 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_347,)
