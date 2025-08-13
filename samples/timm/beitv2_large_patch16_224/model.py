import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_x_: torch.Tensor,
        L_self_modules_patch_embed_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_parameters_cls_token_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_9_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_10_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_11_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_12_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_13_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_14_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_15_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_16_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_17_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_18_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_19_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_20_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_21_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_22_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_23_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_fc_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_fc_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_x_ = L_x_
        l_self_modules_patch_embed_modules_proj_parameters_weight_ = (
            L_self_modules_patch_embed_modules_proj_parameters_weight_
        )
        l_self_modules_patch_embed_modules_proj_parameters_bias_ = (
            L_self_modules_patch_embed_modules_proj_parameters_bias_
        )
        l_self_parameters_cls_token_ = L_self_parameters_cls_token_
        l_self_modules_blocks_modules_0_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_0_parameters_gamma_1_
        )
        l_self_modules_blocks_modules_0_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_0_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_0_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_0_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_0_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_0_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_0_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_blocks_modules_0_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_blocks_modules_0_modules_attn_buffers_relative_position_index_ = L_self_modules_blocks_modules_0_modules_attn_buffers_relative_position_index_
        l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_0_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_0_parameters_gamma_2_
        )
        l_self_modules_blocks_modules_0_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_1_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_1_parameters_gamma_1_
        )
        l_self_modules_blocks_modules_1_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_1_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_1_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_1_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_1_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_1_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_blocks_modules_1_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_blocks_modules_1_modules_attn_buffers_relative_position_index_ = L_self_modules_blocks_modules_1_modules_attn_buffers_relative_position_index_
        l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_1_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_1_parameters_gamma_2_
        )
        l_self_modules_blocks_modules_1_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_2_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_2_parameters_gamma_1_
        )
        l_self_modules_blocks_modules_2_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_2_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_2_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_2_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_2_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_2_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_blocks_modules_2_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_blocks_modules_2_modules_attn_buffers_relative_position_index_ = L_self_modules_blocks_modules_2_modules_attn_buffers_relative_position_index_
        l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_2_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_2_parameters_gamma_2_
        )
        l_self_modules_blocks_modules_2_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_3_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_3_parameters_gamma_1_
        )
        l_self_modules_blocks_modules_3_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_3_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_3_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_3_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_3_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_3_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_blocks_modules_3_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_blocks_modules_3_modules_attn_buffers_relative_position_index_ = L_self_modules_blocks_modules_3_modules_attn_buffers_relative_position_index_
        l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_3_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_3_parameters_gamma_2_
        )
        l_self_modules_blocks_modules_3_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_4_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_4_parameters_gamma_1_
        )
        l_self_modules_blocks_modules_4_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_4_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_4_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_4_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_4_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_4_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_blocks_modules_4_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_blocks_modules_4_modules_attn_buffers_relative_position_index_ = L_self_modules_blocks_modules_4_modules_attn_buffers_relative_position_index_
        l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_4_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_4_parameters_gamma_2_
        )
        l_self_modules_blocks_modules_4_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_5_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_5_parameters_gamma_1_
        )
        l_self_modules_blocks_modules_5_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_5_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_5_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_5_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_5_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_5_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_blocks_modules_5_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_blocks_modules_5_modules_attn_buffers_relative_position_index_ = L_self_modules_blocks_modules_5_modules_attn_buffers_relative_position_index_
        l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_5_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_5_parameters_gamma_2_
        )
        l_self_modules_blocks_modules_5_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_6_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_6_parameters_gamma_1_
        )
        l_self_modules_blocks_modules_6_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_6_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_6_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_6_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_6_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_6_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_blocks_modules_6_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_blocks_modules_6_modules_attn_buffers_relative_position_index_ = L_self_modules_blocks_modules_6_modules_attn_buffers_relative_position_index_
        l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_6_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_6_parameters_gamma_2_
        )
        l_self_modules_blocks_modules_6_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_7_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_7_parameters_gamma_1_
        )
        l_self_modules_blocks_modules_7_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_7_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_7_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_7_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_7_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_7_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_7_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_7_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_7_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_7_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_7_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_blocks_modules_7_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_blocks_modules_7_modules_attn_buffers_relative_position_index_ = L_self_modules_blocks_modules_7_modules_attn_buffers_relative_position_index_
        l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_7_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_7_parameters_gamma_2_
        )
        l_self_modules_blocks_modules_7_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_7_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_7_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_7_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_8_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_8_parameters_gamma_1_
        )
        l_self_modules_blocks_modules_8_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_8_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_8_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_8_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_8_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_8_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_8_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_8_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_8_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_8_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_8_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_blocks_modules_8_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_blocks_modules_8_modules_attn_buffers_relative_position_index_ = L_self_modules_blocks_modules_8_modules_attn_buffers_relative_position_index_
        l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_8_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_8_parameters_gamma_2_
        )
        l_self_modules_blocks_modules_8_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_8_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_8_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_8_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_9_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_9_parameters_gamma_1_
        )
        l_self_modules_blocks_modules_9_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_9_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_9_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_9_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_9_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_9_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_9_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_9_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_9_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_9_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_9_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_blocks_modules_9_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_blocks_modules_9_modules_attn_buffers_relative_position_index_ = L_self_modules_blocks_modules_9_modules_attn_buffers_relative_position_index_
        l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_9_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_9_parameters_gamma_2_
        )
        l_self_modules_blocks_modules_9_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_9_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_9_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_9_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_10_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_10_parameters_gamma_1_
        )
        l_self_modules_blocks_modules_10_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_10_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_10_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_10_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_10_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_10_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_10_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_10_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_10_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_10_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_10_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_blocks_modules_10_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_blocks_modules_10_modules_attn_buffers_relative_position_index_ = L_self_modules_blocks_modules_10_modules_attn_buffers_relative_position_index_
        l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_10_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_10_parameters_gamma_2_
        )
        l_self_modules_blocks_modules_10_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_10_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_10_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_10_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_11_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_11_parameters_gamma_1_
        )
        l_self_modules_blocks_modules_11_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_11_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_11_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_11_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_11_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_11_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_11_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_11_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_11_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_11_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_11_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_blocks_modules_11_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_blocks_modules_11_modules_attn_buffers_relative_position_index_ = L_self_modules_blocks_modules_11_modules_attn_buffers_relative_position_index_
        l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_11_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_11_parameters_gamma_2_
        )
        l_self_modules_blocks_modules_11_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_11_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_11_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_11_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_12_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_12_parameters_gamma_1_
        )
        l_self_modules_blocks_modules_12_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_12_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_12_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_12_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_12_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_12_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_12_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_12_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_12_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_12_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_12_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_blocks_modules_12_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_blocks_modules_12_modules_attn_buffers_relative_position_index_ = L_self_modules_blocks_modules_12_modules_attn_buffers_relative_position_index_
        l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_12_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_12_parameters_gamma_2_
        )
        l_self_modules_blocks_modules_12_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_12_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_12_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_12_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_13_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_13_parameters_gamma_1_
        )
        l_self_modules_blocks_modules_13_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_13_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_13_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_13_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_13_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_13_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_13_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_13_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_13_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_13_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_13_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_blocks_modules_13_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_blocks_modules_13_modules_attn_buffers_relative_position_index_ = L_self_modules_blocks_modules_13_modules_attn_buffers_relative_position_index_
        l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_13_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_13_parameters_gamma_2_
        )
        l_self_modules_blocks_modules_13_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_13_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_13_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_13_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_14_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_14_parameters_gamma_1_
        )
        l_self_modules_blocks_modules_14_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_14_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_14_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_14_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_14_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_14_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_14_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_14_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_14_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_14_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_14_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_blocks_modules_14_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_blocks_modules_14_modules_attn_buffers_relative_position_index_ = L_self_modules_blocks_modules_14_modules_attn_buffers_relative_position_index_
        l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_14_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_14_parameters_gamma_2_
        )
        l_self_modules_blocks_modules_14_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_14_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_14_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_14_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_15_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_15_parameters_gamma_1_
        )
        l_self_modules_blocks_modules_15_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_15_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_15_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_15_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_15_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_15_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_15_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_15_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_15_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_15_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_15_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_blocks_modules_15_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_blocks_modules_15_modules_attn_buffers_relative_position_index_ = L_self_modules_blocks_modules_15_modules_attn_buffers_relative_position_index_
        l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_15_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_15_parameters_gamma_2_
        )
        l_self_modules_blocks_modules_15_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_15_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_15_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_15_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_16_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_16_parameters_gamma_1_
        )
        l_self_modules_blocks_modules_16_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_16_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_16_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_16_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_16_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_16_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_16_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_16_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_16_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_16_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_16_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_blocks_modules_16_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_blocks_modules_16_modules_attn_buffers_relative_position_index_ = L_self_modules_blocks_modules_16_modules_attn_buffers_relative_position_index_
        l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_16_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_16_parameters_gamma_2_
        )
        l_self_modules_blocks_modules_16_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_16_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_16_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_16_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_17_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_17_parameters_gamma_1_
        )
        l_self_modules_blocks_modules_17_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_17_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_17_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_17_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_17_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_17_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_17_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_17_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_17_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_17_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_17_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_blocks_modules_17_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_blocks_modules_17_modules_attn_buffers_relative_position_index_ = L_self_modules_blocks_modules_17_modules_attn_buffers_relative_position_index_
        l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_17_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_17_parameters_gamma_2_
        )
        l_self_modules_blocks_modules_17_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_17_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_17_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_17_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_18_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_18_parameters_gamma_1_
        )
        l_self_modules_blocks_modules_18_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_18_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_18_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_18_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_18_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_18_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_18_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_18_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_18_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_18_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_18_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_blocks_modules_18_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_blocks_modules_18_modules_attn_buffers_relative_position_index_ = L_self_modules_blocks_modules_18_modules_attn_buffers_relative_position_index_
        l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_18_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_18_parameters_gamma_2_
        )
        l_self_modules_blocks_modules_18_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_18_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_18_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_18_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_19_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_19_parameters_gamma_1_
        )
        l_self_modules_blocks_modules_19_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_19_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_19_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_19_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_19_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_19_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_19_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_19_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_19_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_19_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_19_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_blocks_modules_19_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_blocks_modules_19_modules_attn_buffers_relative_position_index_ = L_self_modules_blocks_modules_19_modules_attn_buffers_relative_position_index_
        l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_19_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_19_parameters_gamma_2_
        )
        l_self_modules_blocks_modules_19_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_19_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_19_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_19_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_20_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_20_parameters_gamma_1_
        )
        l_self_modules_blocks_modules_20_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_20_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_20_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_20_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_20_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_20_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_20_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_20_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_20_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_20_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_20_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_blocks_modules_20_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_blocks_modules_20_modules_attn_buffers_relative_position_index_ = L_self_modules_blocks_modules_20_modules_attn_buffers_relative_position_index_
        l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_20_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_20_parameters_gamma_2_
        )
        l_self_modules_blocks_modules_20_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_20_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_20_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_20_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_21_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_21_parameters_gamma_1_
        )
        l_self_modules_blocks_modules_21_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_21_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_21_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_21_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_21_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_21_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_21_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_21_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_21_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_21_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_21_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_blocks_modules_21_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_blocks_modules_21_modules_attn_buffers_relative_position_index_ = L_self_modules_blocks_modules_21_modules_attn_buffers_relative_position_index_
        l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_21_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_21_parameters_gamma_2_
        )
        l_self_modules_blocks_modules_21_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_21_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_21_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_21_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_22_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_22_parameters_gamma_1_
        )
        l_self_modules_blocks_modules_22_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_22_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_22_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_22_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_22_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_22_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_22_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_22_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_22_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_22_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_22_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_blocks_modules_22_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_blocks_modules_22_modules_attn_buffers_relative_position_index_ = L_self_modules_blocks_modules_22_modules_attn_buffers_relative_position_index_
        l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_22_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_22_parameters_gamma_2_
        )
        l_self_modules_blocks_modules_22_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_22_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_22_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_22_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_23_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_23_parameters_gamma_1_
        )
        l_self_modules_blocks_modules_23_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_23_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_23_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_23_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_23_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_23_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_23_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_23_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_23_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_23_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_23_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_blocks_modules_23_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_blocks_modules_23_modules_attn_buffers_relative_position_index_ = L_self_modules_blocks_modules_23_modules_attn_buffers_relative_position_index_
        l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_23_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_23_parameters_gamma_2_
        )
        l_self_modules_blocks_modules_23_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_23_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_23_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_23_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_fc_norm_parameters_weight_ = (
            L_self_modules_fc_norm_parameters_weight_
        )
        l_self_modules_fc_norm_parameters_bias_ = (
            L_self_modules_fc_norm_parameters_bias_
        )
        l_self_modules_head_parameters_weight_ = L_self_modules_head_parameters_weight_
        l_self_modules_head_parameters_bias_ = L_self_modules_head_parameters_bias_
        x = torch.conv2d(
            l_x_,
            l_self_modules_patch_embed_modules_proj_parameters_weight_,
            l_self_modules_patch_embed_modules_proj_parameters_bias_,
            (16, 16),
            (0, 0),
            (1, 1),
            1,
        )
        l_x_ = (
            l_self_modules_patch_embed_modules_proj_parameters_weight_
        ) = l_self_modules_patch_embed_modules_proj_parameters_bias_ = None
        flatten = x.flatten(2)
        x = None
        x_1 = flatten.transpose(1, 2)
        flatten = None
        expand = l_self_parameters_cls_token_.expand(1, -1, -1)
        l_self_parameters_cls_token_ = None
        x_2 = torch.cat((expand, x_1), dim=1)
        expand = x_1 = None
        x_3 = torch.nn.functional.dropout(x_2, 0.0, False, False)
        x_2 = None
        x_4 = torch.nn.functional.layer_norm(
            x_3,
            (1024,),
            l_self_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_0_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_0_modules_norm1_parameters_bias_
        ) = None
        qkv_bias = torch.cat(
            (
                l_self_modules_blocks_modules_0_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_0_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_0_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_0_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_0_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_0_modules_attn_parameters_v_bias_ = None
        qkv = torch._C._nn.linear(
            x_4,
            weight=l_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias,
        )
        x_4 = (
            l_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_
        ) = qkv_bias = None
        reshape = qkv.reshape(1, 197, 3, 16, -1)
        qkv = None
        qkv_1 = reshape.permute(2, 0, 3, 1, 4)
        reshape = None
        unbind = qkv_1.unbind(0)
        qkv_1 = None
        q = unbind[0]
        k = unbind[1]
        v = unbind[2]
        unbind = None
        view = l_self_modules_blocks_modules_0_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_blocks_modules_0_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_3 = l_self_modules_blocks_modules_0_modules_attn_parameters_relative_position_bias_table_[
            view
        ]
        l_self_modules_blocks_modules_0_modules_attn_parameters_relative_position_bias_table_ = (
            view
        ) = None
        relative_position_bias = getitem_3.view(197, 197, -1)
        getitem_3 = None
        permute_1 = relative_position_bias.permute(2, 0, 1)
        relative_position_bias = None
        relative_position_bias_1 = permute_1.contiguous()
        permute_1 = None
        rel_pos_bias = relative_position_bias_1.unsqueeze(0)
        relative_position_bias_1 = None
        x_5 = torch._C._nn.scaled_dot_product_attention(
            q, k, v, attn_mask=rel_pos_bias, dropout_p=0.0
        )
        q = k = v = rel_pos_bias = None
        transpose_1 = x_5.transpose(1, 2)
        x_5 = None
        x_6 = transpose_1.reshape(1, 197, 1024)
        transpose_1 = None
        x_7 = torch._C._nn.linear(
            x_6,
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_6 = (
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_8 = torch.nn.functional.dropout(x_7, 0.0, False, False)
        x_7 = None
        mul = l_self_modules_blocks_modules_0_parameters_gamma_1_ * x_8
        l_self_modules_blocks_modules_0_parameters_gamma_1_ = x_8 = None
        x_9 = x_3 + mul
        x_3 = mul = None
        x_10 = torch.nn.functional.layer_norm(
            x_9,
            (1024,),
            l_self_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_0_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_0_modules_norm2_parameters_bias_
        ) = None
        x_11 = torch._C._nn.linear(
            x_10,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_10 = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_12 = torch._C._nn.gelu(x_11, approximate="none")
        x_11 = None
        x_13 = torch.nn.functional.dropout(x_12, 0.0, False, False)
        x_12 = None
        x_14 = torch._C._nn.linear(
            x_13,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_13 = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_15 = torch.nn.functional.dropout(x_14, 0.0, False, False)
        x_14 = None
        mul_1 = l_self_modules_blocks_modules_0_parameters_gamma_2_ * x_15
        l_self_modules_blocks_modules_0_parameters_gamma_2_ = x_15 = None
        x_16 = x_9 + mul_1
        x_9 = mul_1 = None
        x_17 = torch.nn.functional.layer_norm(
            x_16,
            (1024,),
            l_self_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_1_modules_norm1_parameters_bias_
        ) = None
        qkv_bias_1 = torch.cat(
            (
                l_self_modules_blocks_modules_1_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_1_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_1_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_1_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_1_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_1_modules_attn_parameters_v_bias_ = None
        qkv_2 = torch._C._nn.linear(
            x_17,
            weight=l_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_1,
        )
        x_17 = (
            l_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_
        ) = qkv_bias_1 = None
        reshape_2 = qkv_2.reshape(1, 197, 3, 16, -1)
        qkv_2 = None
        qkv_3 = reshape_2.permute(2, 0, 3, 1, 4)
        reshape_2 = None
        unbind_1 = qkv_3.unbind(0)
        qkv_3 = None
        q_1 = unbind_1[0]
        k_1 = unbind_1[1]
        v_1 = unbind_1[2]
        unbind_1 = None
        view_2 = l_self_modules_blocks_modules_1_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_blocks_modules_1_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_7 = l_self_modules_blocks_modules_1_modules_attn_parameters_relative_position_bias_table_[
            view_2
        ]
        l_self_modules_blocks_modules_1_modules_attn_parameters_relative_position_bias_table_ = (
            view_2
        ) = None
        relative_position_bias_2 = getitem_7.view(197, 197, -1)
        getitem_7 = None
        permute_3 = relative_position_bias_2.permute(2, 0, 1)
        relative_position_bias_2 = None
        relative_position_bias_3 = permute_3.contiguous()
        permute_3 = None
        rel_pos_bias_1 = relative_position_bias_3.unsqueeze(0)
        relative_position_bias_3 = None
        x_18 = torch._C._nn.scaled_dot_product_attention(
            q_1, k_1, v_1, attn_mask=rel_pos_bias_1, dropout_p=0.0
        )
        q_1 = k_1 = v_1 = rel_pos_bias_1 = None
        transpose_2 = x_18.transpose(1, 2)
        x_18 = None
        x_19 = transpose_2.reshape(1, 197, 1024)
        transpose_2 = None
        x_20 = torch._C._nn.linear(
            x_19,
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_19 = (
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_21 = torch.nn.functional.dropout(x_20, 0.0, False, False)
        x_20 = None
        mul_2 = l_self_modules_blocks_modules_1_parameters_gamma_1_ * x_21
        l_self_modules_blocks_modules_1_parameters_gamma_1_ = x_21 = None
        x_22 = x_16 + mul_2
        x_16 = mul_2 = None
        x_23 = torch.nn.functional.layer_norm(
            x_22,
            (1024,),
            l_self_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_1_modules_norm2_parameters_bias_
        ) = None
        x_24 = torch._C._nn.linear(
            x_23,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_23 = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_25 = torch._C._nn.gelu(x_24, approximate="none")
        x_24 = None
        x_26 = torch.nn.functional.dropout(x_25, 0.0, False, False)
        x_25 = None
        x_27 = torch._C._nn.linear(
            x_26,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_26 = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_28 = torch.nn.functional.dropout(x_27, 0.0, False, False)
        x_27 = None
        mul_3 = l_self_modules_blocks_modules_1_parameters_gamma_2_ * x_28
        l_self_modules_blocks_modules_1_parameters_gamma_2_ = x_28 = None
        x_29 = x_22 + mul_3
        x_22 = mul_3 = None
        x_30 = torch.nn.functional.layer_norm(
            x_29,
            (1024,),
            l_self_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_2_modules_norm1_parameters_bias_
        ) = None
        qkv_bias_2 = torch.cat(
            (
                l_self_modules_blocks_modules_2_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_2_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_2_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_2_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_2_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_2_modules_attn_parameters_v_bias_ = None
        qkv_4 = torch._C._nn.linear(
            x_30,
            weight=l_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_2,
        )
        x_30 = (
            l_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_
        ) = qkv_bias_2 = None
        reshape_4 = qkv_4.reshape(1, 197, 3, 16, -1)
        qkv_4 = None
        qkv_5 = reshape_4.permute(2, 0, 3, 1, 4)
        reshape_4 = None
        unbind_2 = qkv_5.unbind(0)
        qkv_5 = None
        q_2 = unbind_2[0]
        k_2 = unbind_2[1]
        v_2 = unbind_2[2]
        unbind_2 = None
        view_4 = l_self_modules_blocks_modules_2_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_blocks_modules_2_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_11 = l_self_modules_blocks_modules_2_modules_attn_parameters_relative_position_bias_table_[
            view_4
        ]
        l_self_modules_blocks_modules_2_modules_attn_parameters_relative_position_bias_table_ = (
            view_4
        ) = None
        relative_position_bias_4 = getitem_11.view(197, 197, -1)
        getitem_11 = None
        permute_5 = relative_position_bias_4.permute(2, 0, 1)
        relative_position_bias_4 = None
        relative_position_bias_5 = permute_5.contiguous()
        permute_5 = None
        rel_pos_bias_2 = relative_position_bias_5.unsqueeze(0)
        relative_position_bias_5 = None
        x_31 = torch._C._nn.scaled_dot_product_attention(
            q_2, k_2, v_2, attn_mask=rel_pos_bias_2, dropout_p=0.0
        )
        q_2 = k_2 = v_2 = rel_pos_bias_2 = None
        transpose_3 = x_31.transpose(1, 2)
        x_31 = None
        x_32 = transpose_3.reshape(1, 197, 1024)
        transpose_3 = None
        x_33 = torch._C._nn.linear(
            x_32,
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_,
        )
        x_32 = (
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_34 = torch.nn.functional.dropout(x_33, 0.0, False, False)
        x_33 = None
        mul_4 = l_self_modules_blocks_modules_2_parameters_gamma_1_ * x_34
        l_self_modules_blocks_modules_2_parameters_gamma_1_ = x_34 = None
        x_35 = x_29 + mul_4
        x_29 = mul_4 = None
        x_36 = torch.nn.functional.layer_norm(
            x_35,
            (1024,),
            l_self_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_2_modules_norm2_parameters_bias_
        ) = None
        x_37 = torch._C._nn.linear(
            x_36,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_36 = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_38 = torch._C._nn.gelu(x_37, approximate="none")
        x_37 = None
        x_39 = torch.nn.functional.dropout(x_38, 0.0, False, False)
        x_38 = None
        x_40 = torch._C._nn.linear(
            x_39,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_39 = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_41 = torch.nn.functional.dropout(x_40, 0.0, False, False)
        x_40 = None
        mul_5 = l_self_modules_blocks_modules_2_parameters_gamma_2_ * x_41
        l_self_modules_blocks_modules_2_parameters_gamma_2_ = x_41 = None
        x_42 = x_35 + mul_5
        x_35 = mul_5 = None
        x_43 = torch.nn.functional.layer_norm(
            x_42,
            (1024,),
            l_self_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_3_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_3_modules_norm1_parameters_bias_
        ) = None
        qkv_bias_3 = torch.cat(
            (
                l_self_modules_blocks_modules_3_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_3_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_3_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_3_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_3_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_3_modules_attn_parameters_v_bias_ = None
        qkv_6 = torch._C._nn.linear(
            x_43,
            weight=l_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_3,
        )
        x_43 = (
            l_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_
        ) = qkv_bias_3 = None
        reshape_6 = qkv_6.reshape(1, 197, 3, 16, -1)
        qkv_6 = None
        qkv_7 = reshape_6.permute(2, 0, 3, 1, 4)
        reshape_6 = None
        unbind_3 = qkv_7.unbind(0)
        qkv_7 = None
        q_3 = unbind_3[0]
        k_3 = unbind_3[1]
        v_3 = unbind_3[2]
        unbind_3 = None
        view_6 = l_self_modules_blocks_modules_3_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_blocks_modules_3_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_15 = l_self_modules_blocks_modules_3_modules_attn_parameters_relative_position_bias_table_[
            view_6
        ]
        l_self_modules_blocks_modules_3_modules_attn_parameters_relative_position_bias_table_ = (
            view_6
        ) = None
        relative_position_bias_6 = getitem_15.view(197, 197, -1)
        getitem_15 = None
        permute_7 = relative_position_bias_6.permute(2, 0, 1)
        relative_position_bias_6 = None
        relative_position_bias_7 = permute_7.contiguous()
        permute_7 = None
        rel_pos_bias_3 = relative_position_bias_7.unsqueeze(0)
        relative_position_bias_7 = None
        x_44 = torch._C._nn.scaled_dot_product_attention(
            q_3, k_3, v_3, attn_mask=rel_pos_bias_3, dropout_p=0.0
        )
        q_3 = k_3 = v_3 = rel_pos_bias_3 = None
        transpose_4 = x_44.transpose(1, 2)
        x_44 = None
        x_45 = transpose_4.reshape(1, 197, 1024)
        transpose_4 = None
        x_46 = torch._C._nn.linear(
            x_45,
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_,
        )
        x_45 = (
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_47 = torch.nn.functional.dropout(x_46, 0.0, False, False)
        x_46 = None
        mul_6 = l_self_modules_blocks_modules_3_parameters_gamma_1_ * x_47
        l_self_modules_blocks_modules_3_parameters_gamma_1_ = x_47 = None
        x_48 = x_42 + mul_6
        x_42 = mul_6 = None
        x_49 = torch.nn.functional.layer_norm(
            x_48,
            (1024,),
            l_self_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_3_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_3_modules_norm2_parameters_bias_
        ) = None
        x_50 = torch._C._nn.linear(
            x_49,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_49 = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_51 = torch._C._nn.gelu(x_50, approximate="none")
        x_50 = None
        x_52 = torch.nn.functional.dropout(x_51, 0.0, False, False)
        x_51 = None
        x_53 = torch._C._nn.linear(
            x_52,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_52 = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_54 = torch.nn.functional.dropout(x_53, 0.0, False, False)
        x_53 = None
        mul_7 = l_self_modules_blocks_modules_3_parameters_gamma_2_ * x_54
        l_self_modules_blocks_modules_3_parameters_gamma_2_ = x_54 = None
        x_55 = x_48 + mul_7
        x_48 = mul_7 = None
        x_56 = torch.nn.functional.layer_norm(
            x_55,
            (1024,),
            l_self_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_4_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_4_modules_norm1_parameters_bias_
        ) = None
        qkv_bias_4 = torch.cat(
            (
                l_self_modules_blocks_modules_4_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_4_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_4_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_4_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_4_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_4_modules_attn_parameters_v_bias_ = None
        qkv_8 = torch._C._nn.linear(
            x_56,
            weight=l_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_4,
        )
        x_56 = (
            l_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_
        ) = qkv_bias_4 = None
        reshape_8 = qkv_8.reshape(1, 197, 3, 16, -1)
        qkv_8 = None
        qkv_9 = reshape_8.permute(2, 0, 3, 1, 4)
        reshape_8 = None
        unbind_4 = qkv_9.unbind(0)
        qkv_9 = None
        q_4 = unbind_4[0]
        k_4 = unbind_4[1]
        v_4 = unbind_4[2]
        unbind_4 = None
        view_8 = l_self_modules_blocks_modules_4_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_blocks_modules_4_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_19 = l_self_modules_blocks_modules_4_modules_attn_parameters_relative_position_bias_table_[
            view_8
        ]
        l_self_modules_blocks_modules_4_modules_attn_parameters_relative_position_bias_table_ = (
            view_8
        ) = None
        relative_position_bias_8 = getitem_19.view(197, 197, -1)
        getitem_19 = None
        permute_9 = relative_position_bias_8.permute(2, 0, 1)
        relative_position_bias_8 = None
        relative_position_bias_9 = permute_9.contiguous()
        permute_9 = None
        rel_pos_bias_4 = relative_position_bias_9.unsqueeze(0)
        relative_position_bias_9 = None
        x_57 = torch._C._nn.scaled_dot_product_attention(
            q_4, k_4, v_4, attn_mask=rel_pos_bias_4, dropout_p=0.0
        )
        q_4 = k_4 = v_4 = rel_pos_bias_4 = None
        transpose_5 = x_57.transpose(1, 2)
        x_57 = None
        x_58 = transpose_5.reshape(1, 197, 1024)
        transpose_5 = None
        x_59 = torch._C._nn.linear(
            x_58,
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_,
        )
        x_58 = (
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_60 = torch.nn.functional.dropout(x_59, 0.0, False, False)
        x_59 = None
        mul_8 = l_self_modules_blocks_modules_4_parameters_gamma_1_ * x_60
        l_self_modules_blocks_modules_4_parameters_gamma_1_ = x_60 = None
        x_61 = x_55 + mul_8
        x_55 = mul_8 = None
        x_62 = torch.nn.functional.layer_norm(
            x_61,
            (1024,),
            l_self_modules_blocks_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_4_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_4_modules_norm2_parameters_bias_
        ) = None
        x_63 = torch._C._nn.linear(
            x_62,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_62 = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_64 = torch._C._nn.gelu(x_63, approximate="none")
        x_63 = None
        x_65 = torch.nn.functional.dropout(x_64, 0.0, False, False)
        x_64 = None
        x_66 = torch._C._nn.linear(
            x_65,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_65 = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_67 = torch.nn.functional.dropout(x_66, 0.0, False, False)
        x_66 = None
        mul_9 = l_self_modules_blocks_modules_4_parameters_gamma_2_ * x_67
        l_self_modules_blocks_modules_4_parameters_gamma_2_ = x_67 = None
        x_68 = x_61 + mul_9
        x_61 = mul_9 = None
        x_69 = torch.nn.functional.layer_norm(
            x_68,
            (1024,),
            l_self_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_5_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_5_modules_norm1_parameters_bias_
        ) = None
        qkv_bias_5 = torch.cat(
            (
                l_self_modules_blocks_modules_5_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_5_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_5_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_5_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_5_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_5_modules_attn_parameters_v_bias_ = None
        qkv_10 = torch._C._nn.linear(
            x_69,
            weight=l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_5,
        )
        x_69 = (
            l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_
        ) = qkv_bias_5 = None
        reshape_10 = qkv_10.reshape(1, 197, 3, 16, -1)
        qkv_10 = None
        qkv_11 = reshape_10.permute(2, 0, 3, 1, 4)
        reshape_10 = None
        unbind_5 = qkv_11.unbind(0)
        qkv_11 = None
        q_5 = unbind_5[0]
        k_5 = unbind_5[1]
        v_5 = unbind_5[2]
        unbind_5 = None
        view_10 = l_self_modules_blocks_modules_5_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_blocks_modules_5_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_23 = l_self_modules_blocks_modules_5_modules_attn_parameters_relative_position_bias_table_[
            view_10
        ]
        l_self_modules_blocks_modules_5_modules_attn_parameters_relative_position_bias_table_ = (
            view_10
        ) = None
        relative_position_bias_10 = getitem_23.view(197, 197, -1)
        getitem_23 = None
        permute_11 = relative_position_bias_10.permute(2, 0, 1)
        relative_position_bias_10 = None
        relative_position_bias_11 = permute_11.contiguous()
        permute_11 = None
        rel_pos_bias_5 = relative_position_bias_11.unsqueeze(0)
        relative_position_bias_11 = None
        x_70 = torch._C._nn.scaled_dot_product_attention(
            q_5, k_5, v_5, attn_mask=rel_pos_bias_5, dropout_p=0.0
        )
        q_5 = k_5 = v_5 = rel_pos_bias_5 = None
        transpose_6 = x_70.transpose(1, 2)
        x_70 = None
        x_71 = transpose_6.reshape(1, 197, 1024)
        transpose_6 = None
        x_72 = torch._C._nn.linear(
            x_71,
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_,
        )
        x_71 = (
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_73 = torch.nn.functional.dropout(x_72, 0.0, False, False)
        x_72 = None
        mul_10 = l_self_modules_blocks_modules_5_parameters_gamma_1_ * x_73
        l_self_modules_blocks_modules_5_parameters_gamma_1_ = x_73 = None
        x_74 = x_68 + mul_10
        x_68 = mul_10 = None
        x_75 = torch.nn.functional.layer_norm(
            x_74,
            (1024,),
            l_self_modules_blocks_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_5_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_5_modules_norm2_parameters_bias_
        ) = None
        x_76 = torch._C._nn.linear(
            x_75,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_75 = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_77 = torch._C._nn.gelu(x_76, approximate="none")
        x_76 = None
        x_78 = torch.nn.functional.dropout(x_77, 0.0, False, False)
        x_77 = None
        x_79 = torch._C._nn.linear(
            x_78,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_78 = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_80 = torch.nn.functional.dropout(x_79, 0.0, False, False)
        x_79 = None
        mul_11 = l_self_modules_blocks_modules_5_parameters_gamma_2_ * x_80
        l_self_modules_blocks_modules_5_parameters_gamma_2_ = x_80 = None
        x_81 = x_74 + mul_11
        x_74 = mul_11 = None
        x_82 = torch.nn.functional.layer_norm(
            x_81,
            (1024,),
            l_self_modules_blocks_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_6_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_6_modules_norm1_parameters_bias_
        ) = None
        qkv_bias_6 = torch.cat(
            (
                l_self_modules_blocks_modules_6_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_6_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_6_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_6_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_6_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_6_modules_attn_parameters_v_bias_ = None
        qkv_12 = torch._C._nn.linear(
            x_82,
            weight=l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_6,
        )
        x_82 = (
            l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_
        ) = qkv_bias_6 = None
        reshape_12 = qkv_12.reshape(1, 197, 3, 16, -1)
        qkv_12 = None
        qkv_13 = reshape_12.permute(2, 0, 3, 1, 4)
        reshape_12 = None
        unbind_6 = qkv_13.unbind(0)
        qkv_13 = None
        q_6 = unbind_6[0]
        k_6 = unbind_6[1]
        v_6 = unbind_6[2]
        unbind_6 = None
        view_12 = l_self_modules_blocks_modules_6_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_blocks_modules_6_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_27 = l_self_modules_blocks_modules_6_modules_attn_parameters_relative_position_bias_table_[
            view_12
        ]
        l_self_modules_blocks_modules_6_modules_attn_parameters_relative_position_bias_table_ = (
            view_12
        ) = None
        relative_position_bias_12 = getitem_27.view(197, 197, -1)
        getitem_27 = None
        permute_13 = relative_position_bias_12.permute(2, 0, 1)
        relative_position_bias_12 = None
        relative_position_bias_13 = permute_13.contiguous()
        permute_13 = None
        rel_pos_bias_6 = relative_position_bias_13.unsqueeze(0)
        relative_position_bias_13 = None
        x_83 = torch._C._nn.scaled_dot_product_attention(
            q_6, k_6, v_6, attn_mask=rel_pos_bias_6, dropout_p=0.0
        )
        q_6 = k_6 = v_6 = rel_pos_bias_6 = None
        transpose_7 = x_83.transpose(1, 2)
        x_83 = None
        x_84 = transpose_7.reshape(1, 197, 1024)
        transpose_7 = None
        x_85 = torch._C._nn.linear(
            x_84,
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_,
        )
        x_84 = (
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_86 = torch.nn.functional.dropout(x_85, 0.0, False, False)
        x_85 = None
        mul_12 = l_self_modules_blocks_modules_6_parameters_gamma_1_ * x_86
        l_self_modules_blocks_modules_6_parameters_gamma_1_ = x_86 = None
        x_87 = x_81 + mul_12
        x_81 = mul_12 = None
        x_88 = torch.nn.functional.layer_norm(
            x_87,
            (1024,),
            l_self_modules_blocks_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_6_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_6_modules_norm2_parameters_bias_
        ) = None
        x_89 = torch._C._nn.linear(
            x_88,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_88 = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_90 = torch._C._nn.gelu(x_89, approximate="none")
        x_89 = None
        x_91 = torch.nn.functional.dropout(x_90, 0.0, False, False)
        x_90 = None
        x_92 = torch._C._nn.linear(
            x_91,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_91 = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_93 = torch.nn.functional.dropout(x_92, 0.0, False, False)
        x_92 = None
        mul_13 = l_self_modules_blocks_modules_6_parameters_gamma_2_ * x_93
        l_self_modules_blocks_modules_6_parameters_gamma_2_ = x_93 = None
        x_94 = x_87 + mul_13
        x_87 = mul_13 = None
        x_95 = torch.nn.functional.layer_norm(
            x_94,
            (1024,),
            l_self_modules_blocks_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_7_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_7_modules_norm1_parameters_bias_
        ) = None
        qkv_bias_7 = torch.cat(
            (
                l_self_modules_blocks_modules_7_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_7_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_7_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_7_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_7_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_7_modules_attn_parameters_v_bias_ = None
        qkv_14 = torch._C._nn.linear(
            x_95,
            weight=l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_7,
        )
        x_95 = (
            l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_
        ) = qkv_bias_7 = None
        reshape_14 = qkv_14.reshape(1, 197, 3, 16, -1)
        qkv_14 = None
        qkv_15 = reshape_14.permute(2, 0, 3, 1, 4)
        reshape_14 = None
        unbind_7 = qkv_15.unbind(0)
        qkv_15 = None
        q_7 = unbind_7[0]
        k_7 = unbind_7[1]
        v_7 = unbind_7[2]
        unbind_7 = None
        view_14 = l_self_modules_blocks_modules_7_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_blocks_modules_7_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_31 = l_self_modules_blocks_modules_7_modules_attn_parameters_relative_position_bias_table_[
            view_14
        ]
        l_self_modules_blocks_modules_7_modules_attn_parameters_relative_position_bias_table_ = (
            view_14
        ) = None
        relative_position_bias_14 = getitem_31.view(197, 197, -1)
        getitem_31 = None
        permute_15 = relative_position_bias_14.permute(2, 0, 1)
        relative_position_bias_14 = None
        relative_position_bias_15 = permute_15.contiguous()
        permute_15 = None
        rel_pos_bias_7 = relative_position_bias_15.unsqueeze(0)
        relative_position_bias_15 = None
        x_96 = torch._C._nn.scaled_dot_product_attention(
            q_7, k_7, v_7, attn_mask=rel_pos_bias_7, dropout_p=0.0
        )
        q_7 = k_7 = v_7 = rel_pos_bias_7 = None
        transpose_8 = x_96.transpose(1, 2)
        x_96 = None
        x_97 = transpose_8.reshape(1, 197, 1024)
        transpose_8 = None
        x_98 = torch._C._nn.linear(
            x_97,
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_,
        )
        x_97 = (
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_99 = torch.nn.functional.dropout(x_98, 0.0, False, False)
        x_98 = None
        mul_14 = l_self_modules_blocks_modules_7_parameters_gamma_1_ * x_99
        l_self_modules_blocks_modules_7_parameters_gamma_1_ = x_99 = None
        x_100 = x_94 + mul_14
        x_94 = mul_14 = None
        x_101 = torch.nn.functional.layer_norm(
            x_100,
            (1024,),
            l_self_modules_blocks_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_7_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_7_modules_norm2_parameters_bias_
        ) = None
        x_102 = torch._C._nn.linear(
            x_101,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_101 = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_103 = torch._C._nn.gelu(x_102, approximate="none")
        x_102 = None
        x_104 = torch.nn.functional.dropout(x_103, 0.0, False, False)
        x_103 = None
        x_105 = torch._C._nn.linear(
            x_104,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_104 = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_106 = torch.nn.functional.dropout(x_105, 0.0, False, False)
        x_105 = None
        mul_15 = l_self_modules_blocks_modules_7_parameters_gamma_2_ * x_106
        l_self_modules_blocks_modules_7_parameters_gamma_2_ = x_106 = None
        x_107 = x_100 + mul_15
        x_100 = mul_15 = None
        x_108 = torch.nn.functional.layer_norm(
            x_107,
            (1024,),
            l_self_modules_blocks_modules_8_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_8_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_8_modules_norm1_parameters_bias_
        ) = None
        qkv_bias_8 = torch.cat(
            (
                l_self_modules_blocks_modules_8_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_8_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_8_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_8_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_8_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_8_modules_attn_parameters_v_bias_ = None
        qkv_16 = torch._C._nn.linear(
            x_108,
            weight=l_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_8,
        )
        x_108 = (
            l_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_
        ) = qkv_bias_8 = None
        reshape_16 = qkv_16.reshape(1, 197, 3, 16, -1)
        qkv_16 = None
        qkv_17 = reshape_16.permute(2, 0, 3, 1, 4)
        reshape_16 = None
        unbind_8 = qkv_17.unbind(0)
        qkv_17 = None
        q_8 = unbind_8[0]
        k_8 = unbind_8[1]
        v_8 = unbind_8[2]
        unbind_8 = None
        view_16 = l_self_modules_blocks_modules_8_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_blocks_modules_8_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_35 = l_self_modules_blocks_modules_8_modules_attn_parameters_relative_position_bias_table_[
            view_16
        ]
        l_self_modules_blocks_modules_8_modules_attn_parameters_relative_position_bias_table_ = (
            view_16
        ) = None
        relative_position_bias_16 = getitem_35.view(197, 197, -1)
        getitem_35 = None
        permute_17 = relative_position_bias_16.permute(2, 0, 1)
        relative_position_bias_16 = None
        relative_position_bias_17 = permute_17.contiguous()
        permute_17 = None
        rel_pos_bias_8 = relative_position_bias_17.unsqueeze(0)
        relative_position_bias_17 = None
        x_109 = torch._C._nn.scaled_dot_product_attention(
            q_8, k_8, v_8, attn_mask=rel_pos_bias_8, dropout_p=0.0
        )
        q_8 = k_8 = v_8 = rel_pos_bias_8 = None
        transpose_9 = x_109.transpose(1, 2)
        x_109 = None
        x_110 = transpose_9.reshape(1, 197, 1024)
        transpose_9 = None
        x_111 = torch._C._nn.linear(
            x_110,
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_,
        )
        x_110 = (
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_112 = torch.nn.functional.dropout(x_111, 0.0, False, False)
        x_111 = None
        mul_16 = l_self_modules_blocks_modules_8_parameters_gamma_1_ * x_112
        l_self_modules_blocks_modules_8_parameters_gamma_1_ = x_112 = None
        x_113 = x_107 + mul_16
        x_107 = mul_16 = None
        x_114 = torch.nn.functional.layer_norm(
            x_113,
            (1024,),
            l_self_modules_blocks_modules_8_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_8_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_8_modules_norm2_parameters_bias_
        ) = None
        x_115 = torch._C._nn.linear(
            x_114,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_114 = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_116 = torch._C._nn.gelu(x_115, approximate="none")
        x_115 = None
        x_117 = torch.nn.functional.dropout(x_116, 0.0, False, False)
        x_116 = None
        x_118 = torch._C._nn.linear(
            x_117,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_117 = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_119 = torch.nn.functional.dropout(x_118, 0.0, False, False)
        x_118 = None
        mul_17 = l_self_modules_blocks_modules_8_parameters_gamma_2_ * x_119
        l_self_modules_blocks_modules_8_parameters_gamma_2_ = x_119 = None
        x_120 = x_113 + mul_17
        x_113 = mul_17 = None
        x_121 = torch.nn.functional.layer_norm(
            x_120,
            (1024,),
            l_self_modules_blocks_modules_9_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_9_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_9_modules_norm1_parameters_bias_
        ) = None
        qkv_bias_9 = torch.cat(
            (
                l_self_modules_blocks_modules_9_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_9_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_9_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_9_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_9_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_9_modules_attn_parameters_v_bias_ = None
        qkv_18 = torch._C._nn.linear(
            x_121,
            weight=l_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_9,
        )
        x_121 = (
            l_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_
        ) = qkv_bias_9 = None
        reshape_18 = qkv_18.reshape(1, 197, 3, 16, -1)
        qkv_18 = None
        qkv_19 = reshape_18.permute(2, 0, 3, 1, 4)
        reshape_18 = None
        unbind_9 = qkv_19.unbind(0)
        qkv_19 = None
        q_9 = unbind_9[0]
        k_9 = unbind_9[1]
        v_9 = unbind_9[2]
        unbind_9 = None
        view_18 = l_self_modules_blocks_modules_9_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_blocks_modules_9_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_39 = l_self_modules_blocks_modules_9_modules_attn_parameters_relative_position_bias_table_[
            view_18
        ]
        l_self_modules_blocks_modules_9_modules_attn_parameters_relative_position_bias_table_ = (
            view_18
        ) = None
        relative_position_bias_18 = getitem_39.view(197, 197, -1)
        getitem_39 = None
        permute_19 = relative_position_bias_18.permute(2, 0, 1)
        relative_position_bias_18 = None
        relative_position_bias_19 = permute_19.contiguous()
        permute_19 = None
        rel_pos_bias_9 = relative_position_bias_19.unsqueeze(0)
        relative_position_bias_19 = None
        x_122 = torch._C._nn.scaled_dot_product_attention(
            q_9, k_9, v_9, attn_mask=rel_pos_bias_9, dropout_p=0.0
        )
        q_9 = k_9 = v_9 = rel_pos_bias_9 = None
        transpose_10 = x_122.transpose(1, 2)
        x_122 = None
        x_123 = transpose_10.reshape(1, 197, 1024)
        transpose_10 = None
        x_124 = torch._C._nn.linear(
            x_123,
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_,
        )
        x_123 = (
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_125 = torch.nn.functional.dropout(x_124, 0.0, False, False)
        x_124 = None
        mul_18 = l_self_modules_blocks_modules_9_parameters_gamma_1_ * x_125
        l_self_modules_blocks_modules_9_parameters_gamma_1_ = x_125 = None
        x_126 = x_120 + mul_18
        x_120 = mul_18 = None
        x_127 = torch.nn.functional.layer_norm(
            x_126,
            (1024,),
            l_self_modules_blocks_modules_9_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_9_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_9_modules_norm2_parameters_bias_
        ) = None
        x_128 = torch._C._nn.linear(
            x_127,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_127 = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_129 = torch._C._nn.gelu(x_128, approximate="none")
        x_128 = None
        x_130 = torch.nn.functional.dropout(x_129, 0.0, False, False)
        x_129 = None
        x_131 = torch._C._nn.linear(
            x_130,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_130 = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_132 = torch.nn.functional.dropout(x_131, 0.0, False, False)
        x_131 = None
        mul_19 = l_self_modules_blocks_modules_9_parameters_gamma_2_ * x_132
        l_self_modules_blocks_modules_9_parameters_gamma_2_ = x_132 = None
        x_133 = x_126 + mul_19
        x_126 = mul_19 = None
        x_134 = torch.nn.functional.layer_norm(
            x_133,
            (1024,),
            l_self_modules_blocks_modules_10_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_10_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_norm1_parameters_bias_
        ) = None
        qkv_bias_10 = torch.cat(
            (
                l_self_modules_blocks_modules_10_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_10_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_10_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_10_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_10_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_10_modules_attn_parameters_v_bias_ = None
        qkv_20 = torch._C._nn.linear(
            x_134,
            weight=l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_10,
        )
        x_134 = (
            l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_
        ) = qkv_bias_10 = None
        reshape_20 = qkv_20.reshape(1, 197, 3, 16, -1)
        qkv_20 = None
        qkv_21 = reshape_20.permute(2, 0, 3, 1, 4)
        reshape_20 = None
        unbind_10 = qkv_21.unbind(0)
        qkv_21 = None
        q_10 = unbind_10[0]
        k_10 = unbind_10[1]
        v_10 = unbind_10[2]
        unbind_10 = None
        view_20 = l_self_modules_blocks_modules_10_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_blocks_modules_10_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_43 = l_self_modules_blocks_modules_10_modules_attn_parameters_relative_position_bias_table_[
            view_20
        ]
        l_self_modules_blocks_modules_10_modules_attn_parameters_relative_position_bias_table_ = (
            view_20
        ) = None
        relative_position_bias_20 = getitem_43.view(197, 197, -1)
        getitem_43 = None
        permute_21 = relative_position_bias_20.permute(2, 0, 1)
        relative_position_bias_20 = None
        relative_position_bias_21 = permute_21.contiguous()
        permute_21 = None
        rel_pos_bias_10 = relative_position_bias_21.unsqueeze(0)
        relative_position_bias_21 = None
        x_135 = torch._C._nn.scaled_dot_product_attention(
            q_10, k_10, v_10, attn_mask=rel_pos_bias_10, dropout_p=0.0
        )
        q_10 = k_10 = v_10 = rel_pos_bias_10 = None
        transpose_11 = x_135.transpose(1, 2)
        x_135 = None
        x_136 = transpose_11.reshape(1, 197, 1024)
        transpose_11 = None
        x_137 = torch._C._nn.linear(
            x_136,
            l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_,
        )
        x_136 = l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_138 = torch.nn.functional.dropout(x_137, 0.0, False, False)
        x_137 = None
        mul_20 = l_self_modules_blocks_modules_10_parameters_gamma_1_ * x_138
        l_self_modules_blocks_modules_10_parameters_gamma_1_ = x_138 = None
        x_139 = x_133 + mul_20
        x_133 = mul_20 = None
        x_140 = torch.nn.functional.layer_norm(
            x_139,
            (1024,),
            l_self_modules_blocks_modules_10_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_10_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_norm2_parameters_bias_
        ) = None
        x_141 = torch._C._nn.linear(
            x_140,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_140 = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_142 = torch._C._nn.gelu(x_141, approximate="none")
        x_141 = None
        x_143 = torch.nn.functional.dropout(x_142, 0.0, False, False)
        x_142 = None
        x_144 = torch._C._nn.linear(
            x_143,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_143 = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_145 = torch.nn.functional.dropout(x_144, 0.0, False, False)
        x_144 = None
        mul_21 = l_self_modules_blocks_modules_10_parameters_gamma_2_ * x_145
        l_self_modules_blocks_modules_10_parameters_gamma_2_ = x_145 = None
        x_146 = x_139 + mul_21
        x_139 = mul_21 = None
        x_147 = torch.nn.functional.layer_norm(
            x_146,
            (1024,),
            l_self_modules_blocks_modules_11_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_11_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_norm1_parameters_bias_
        ) = None
        qkv_bias_11 = torch.cat(
            (
                l_self_modules_blocks_modules_11_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_11_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_11_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_11_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_11_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_11_modules_attn_parameters_v_bias_ = None
        qkv_22 = torch._C._nn.linear(
            x_147,
            weight=l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_11,
        )
        x_147 = (
            l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_
        ) = qkv_bias_11 = None
        reshape_22 = qkv_22.reshape(1, 197, 3, 16, -1)
        qkv_22 = None
        qkv_23 = reshape_22.permute(2, 0, 3, 1, 4)
        reshape_22 = None
        unbind_11 = qkv_23.unbind(0)
        qkv_23 = None
        q_11 = unbind_11[0]
        k_11 = unbind_11[1]
        v_11 = unbind_11[2]
        unbind_11 = None
        view_22 = l_self_modules_blocks_modules_11_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_blocks_modules_11_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_47 = l_self_modules_blocks_modules_11_modules_attn_parameters_relative_position_bias_table_[
            view_22
        ]
        l_self_modules_blocks_modules_11_modules_attn_parameters_relative_position_bias_table_ = (
            view_22
        ) = None
        relative_position_bias_22 = getitem_47.view(197, 197, -1)
        getitem_47 = None
        permute_23 = relative_position_bias_22.permute(2, 0, 1)
        relative_position_bias_22 = None
        relative_position_bias_23 = permute_23.contiguous()
        permute_23 = None
        rel_pos_bias_11 = relative_position_bias_23.unsqueeze(0)
        relative_position_bias_23 = None
        x_148 = torch._C._nn.scaled_dot_product_attention(
            q_11, k_11, v_11, attn_mask=rel_pos_bias_11, dropout_p=0.0
        )
        q_11 = k_11 = v_11 = rel_pos_bias_11 = None
        transpose_12 = x_148.transpose(1, 2)
        x_148 = None
        x_149 = transpose_12.reshape(1, 197, 1024)
        transpose_12 = None
        x_150 = torch._C._nn.linear(
            x_149,
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_,
        )
        x_149 = l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_151 = torch.nn.functional.dropout(x_150, 0.0, False, False)
        x_150 = None
        mul_22 = l_self_modules_blocks_modules_11_parameters_gamma_1_ * x_151
        l_self_modules_blocks_modules_11_parameters_gamma_1_ = x_151 = None
        x_152 = x_146 + mul_22
        x_146 = mul_22 = None
        x_153 = torch.nn.functional.layer_norm(
            x_152,
            (1024,),
            l_self_modules_blocks_modules_11_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_11_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_norm2_parameters_bias_
        ) = None
        x_154 = torch._C._nn.linear(
            x_153,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_153 = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_155 = torch._C._nn.gelu(x_154, approximate="none")
        x_154 = None
        x_156 = torch.nn.functional.dropout(x_155, 0.0, False, False)
        x_155 = None
        x_157 = torch._C._nn.linear(
            x_156,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_156 = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_158 = torch.nn.functional.dropout(x_157, 0.0, False, False)
        x_157 = None
        mul_23 = l_self_modules_blocks_modules_11_parameters_gamma_2_ * x_158
        l_self_modules_blocks_modules_11_parameters_gamma_2_ = x_158 = None
        x_159 = x_152 + mul_23
        x_152 = mul_23 = None
        x_160 = torch.nn.functional.layer_norm(
            x_159,
            (1024,),
            l_self_modules_blocks_modules_12_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_12_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_12_modules_norm1_parameters_bias_
        ) = None
        qkv_bias_12 = torch.cat(
            (
                l_self_modules_blocks_modules_12_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_12_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_12_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_12_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_12_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_12_modules_attn_parameters_v_bias_ = None
        qkv_24 = torch._C._nn.linear(
            x_160,
            weight=l_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_12,
        )
        x_160 = (
            l_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_
        ) = qkv_bias_12 = None
        reshape_24 = qkv_24.reshape(1, 197, 3, 16, -1)
        qkv_24 = None
        qkv_25 = reshape_24.permute(2, 0, 3, 1, 4)
        reshape_24 = None
        unbind_12 = qkv_25.unbind(0)
        qkv_25 = None
        q_12 = unbind_12[0]
        k_12 = unbind_12[1]
        v_12 = unbind_12[2]
        unbind_12 = None
        view_24 = l_self_modules_blocks_modules_12_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_blocks_modules_12_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_51 = l_self_modules_blocks_modules_12_modules_attn_parameters_relative_position_bias_table_[
            view_24
        ]
        l_self_modules_blocks_modules_12_modules_attn_parameters_relative_position_bias_table_ = (
            view_24
        ) = None
        relative_position_bias_24 = getitem_51.view(197, 197, -1)
        getitem_51 = None
        permute_25 = relative_position_bias_24.permute(2, 0, 1)
        relative_position_bias_24 = None
        relative_position_bias_25 = permute_25.contiguous()
        permute_25 = None
        rel_pos_bias_12 = relative_position_bias_25.unsqueeze(0)
        relative_position_bias_25 = None
        x_161 = torch._C._nn.scaled_dot_product_attention(
            q_12, k_12, v_12, attn_mask=rel_pos_bias_12, dropout_p=0.0
        )
        q_12 = k_12 = v_12 = rel_pos_bias_12 = None
        transpose_13 = x_161.transpose(1, 2)
        x_161 = None
        x_162 = transpose_13.reshape(1, 197, 1024)
        transpose_13 = None
        x_163 = torch._C._nn.linear(
            x_162,
            l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_,
        )
        x_162 = l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_164 = torch.nn.functional.dropout(x_163, 0.0, False, False)
        x_163 = None
        mul_24 = l_self_modules_blocks_modules_12_parameters_gamma_1_ * x_164
        l_self_modules_blocks_modules_12_parameters_gamma_1_ = x_164 = None
        x_165 = x_159 + mul_24
        x_159 = mul_24 = None
        x_166 = torch.nn.functional.layer_norm(
            x_165,
            (1024,),
            l_self_modules_blocks_modules_12_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_12_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_12_modules_norm2_parameters_bias_
        ) = None
        x_167 = torch._C._nn.linear(
            x_166,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_166 = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_168 = torch._C._nn.gelu(x_167, approximate="none")
        x_167 = None
        x_169 = torch.nn.functional.dropout(x_168, 0.0, False, False)
        x_168 = None
        x_170 = torch._C._nn.linear(
            x_169,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_169 = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_171 = torch.nn.functional.dropout(x_170, 0.0, False, False)
        x_170 = None
        mul_25 = l_self_modules_blocks_modules_12_parameters_gamma_2_ * x_171
        l_self_modules_blocks_modules_12_parameters_gamma_2_ = x_171 = None
        x_172 = x_165 + mul_25
        x_165 = mul_25 = None
        x_173 = torch.nn.functional.layer_norm(
            x_172,
            (1024,),
            l_self_modules_blocks_modules_13_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_13_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_13_modules_norm1_parameters_bias_
        ) = None
        qkv_bias_13 = torch.cat(
            (
                l_self_modules_blocks_modules_13_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_13_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_13_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_13_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_13_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_13_modules_attn_parameters_v_bias_ = None
        qkv_26 = torch._C._nn.linear(
            x_173,
            weight=l_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_13,
        )
        x_173 = (
            l_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_
        ) = qkv_bias_13 = None
        reshape_26 = qkv_26.reshape(1, 197, 3, 16, -1)
        qkv_26 = None
        qkv_27 = reshape_26.permute(2, 0, 3, 1, 4)
        reshape_26 = None
        unbind_13 = qkv_27.unbind(0)
        qkv_27 = None
        q_13 = unbind_13[0]
        k_13 = unbind_13[1]
        v_13 = unbind_13[2]
        unbind_13 = None
        view_26 = l_self_modules_blocks_modules_13_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_blocks_modules_13_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_55 = l_self_modules_blocks_modules_13_modules_attn_parameters_relative_position_bias_table_[
            view_26
        ]
        l_self_modules_blocks_modules_13_modules_attn_parameters_relative_position_bias_table_ = (
            view_26
        ) = None
        relative_position_bias_26 = getitem_55.view(197, 197, -1)
        getitem_55 = None
        permute_27 = relative_position_bias_26.permute(2, 0, 1)
        relative_position_bias_26 = None
        relative_position_bias_27 = permute_27.contiguous()
        permute_27 = None
        rel_pos_bias_13 = relative_position_bias_27.unsqueeze(0)
        relative_position_bias_27 = None
        x_174 = torch._C._nn.scaled_dot_product_attention(
            q_13, k_13, v_13, attn_mask=rel_pos_bias_13, dropout_p=0.0
        )
        q_13 = k_13 = v_13 = rel_pos_bias_13 = None
        transpose_14 = x_174.transpose(1, 2)
        x_174 = None
        x_175 = transpose_14.reshape(1, 197, 1024)
        transpose_14 = None
        x_176 = torch._C._nn.linear(
            x_175,
            l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_,
        )
        x_175 = l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_177 = torch.nn.functional.dropout(x_176, 0.0, False, False)
        x_176 = None
        mul_26 = l_self_modules_blocks_modules_13_parameters_gamma_1_ * x_177
        l_self_modules_blocks_modules_13_parameters_gamma_1_ = x_177 = None
        x_178 = x_172 + mul_26
        x_172 = mul_26 = None
        x_179 = torch.nn.functional.layer_norm(
            x_178,
            (1024,),
            l_self_modules_blocks_modules_13_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_13_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_13_modules_norm2_parameters_bias_
        ) = None
        x_180 = torch._C._nn.linear(
            x_179,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_179 = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_181 = torch._C._nn.gelu(x_180, approximate="none")
        x_180 = None
        x_182 = torch.nn.functional.dropout(x_181, 0.0, False, False)
        x_181 = None
        x_183 = torch._C._nn.linear(
            x_182,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_182 = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_184 = torch.nn.functional.dropout(x_183, 0.0, False, False)
        x_183 = None
        mul_27 = l_self_modules_blocks_modules_13_parameters_gamma_2_ * x_184
        l_self_modules_blocks_modules_13_parameters_gamma_2_ = x_184 = None
        x_185 = x_178 + mul_27
        x_178 = mul_27 = None
        x_186 = torch.nn.functional.layer_norm(
            x_185,
            (1024,),
            l_self_modules_blocks_modules_14_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_14_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_14_modules_norm1_parameters_bias_
        ) = None
        qkv_bias_14 = torch.cat(
            (
                l_self_modules_blocks_modules_14_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_14_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_14_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_14_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_14_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_14_modules_attn_parameters_v_bias_ = None
        qkv_28 = torch._C._nn.linear(
            x_186,
            weight=l_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_14,
        )
        x_186 = (
            l_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_
        ) = qkv_bias_14 = None
        reshape_28 = qkv_28.reshape(1, 197, 3, 16, -1)
        qkv_28 = None
        qkv_29 = reshape_28.permute(2, 0, 3, 1, 4)
        reshape_28 = None
        unbind_14 = qkv_29.unbind(0)
        qkv_29 = None
        q_14 = unbind_14[0]
        k_14 = unbind_14[1]
        v_14 = unbind_14[2]
        unbind_14 = None
        view_28 = l_self_modules_blocks_modules_14_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_blocks_modules_14_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_59 = l_self_modules_blocks_modules_14_modules_attn_parameters_relative_position_bias_table_[
            view_28
        ]
        l_self_modules_blocks_modules_14_modules_attn_parameters_relative_position_bias_table_ = (
            view_28
        ) = None
        relative_position_bias_28 = getitem_59.view(197, 197, -1)
        getitem_59 = None
        permute_29 = relative_position_bias_28.permute(2, 0, 1)
        relative_position_bias_28 = None
        relative_position_bias_29 = permute_29.contiguous()
        permute_29 = None
        rel_pos_bias_14 = relative_position_bias_29.unsqueeze(0)
        relative_position_bias_29 = None
        x_187 = torch._C._nn.scaled_dot_product_attention(
            q_14, k_14, v_14, attn_mask=rel_pos_bias_14, dropout_p=0.0
        )
        q_14 = k_14 = v_14 = rel_pos_bias_14 = None
        transpose_15 = x_187.transpose(1, 2)
        x_187 = None
        x_188 = transpose_15.reshape(1, 197, 1024)
        transpose_15 = None
        x_189 = torch._C._nn.linear(
            x_188,
            l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_,
        )
        x_188 = l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_190 = torch.nn.functional.dropout(x_189, 0.0, False, False)
        x_189 = None
        mul_28 = l_self_modules_blocks_modules_14_parameters_gamma_1_ * x_190
        l_self_modules_blocks_modules_14_parameters_gamma_1_ = x_190 = None
        x_191 = x_185 + mul_28
        x_185 = mul_28 = None
        x_192 = torch.nn.functional.layer_norm(
            x_191,
            (1024,),
            l_self_modules_blocks_modules_14_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_14_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_14_modules_norm2_parameters_bias_
        ) = None
        x_193 = torch._C._nn.linear(
            x_192,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_192 = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_194 = torch._C._nn.gelu(x_193, approximate="none")
        x_193 = None
        x_195 = torch.nn.functional.dropout(x_194, 0.0, False, False)
        x_194 = None
        x_196 = torch._C._nn.linear(
            x_195,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_195 = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_197 = torch.nn.functional.dropout(x_196, 0.0, False, False)
        x_196 = None
        mul_29 = l_self_modules_blocks_modules_14_parameters_gamma_2_ * x_197
        l_self_modules_blocks_modules_14_parameters_gamma_2_ = x_197 = None
        x_198 = x_191 + mul_29
        x_191 = mul_29 = None
        x_199 = torch.nn.functional.layer_norm(
            x_198,
            (1024,),
            l_self_modules_blocks_modules_15_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_15_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_15_modules_norm1_parameters_bias_
        ) = None
        qkv_bias_15 = torch.cat(
            (
                l_self_modules_blocks_modules_15_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_15_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_15_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_15_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_15_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_15_modules_attn_parameters_v_bias_ = None
        qkv_30 = torch._C._nn.linear(
            x_199,
            weight=l_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_15,
        )
        x_199 = (
            l_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_
        ) = qkv_bias_15 = None
        reshape_30 = qkv_30.reshape(1, 197, 3, 16, -1)
        qkv_30 = None
        qkv_31 = reshape_30.permute(2, 0, 3, 1, 4)
        reshape_30 = None
        unbind_15 = qkv_31.unbind(0)
        qkv_31 = None
        q_15 = unbind_15[0]
        k_15 = unbind_15[1]
        v_15 = unbind_15[2]
        unbind_15 = None
        view_30 = l_self_modules_blocks_modules_15_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_blocks_modules_15_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_63 = l_self_modules_blocks_modules_15_modules_attn_parameters_relative_position_bias_table_[
            view_30
        ]
        l_self_modules_blocks_modules_15_modules_attn_parameters_relative_position_bias_table_ = (
            view_30
        ) = None
        relative_position_bias_30 = getitem_63.view(197, 197, -1)
        getitem_63 = None
        permute_31 = relative_position_bias_30.permute(2, 0, 1)
        relative_position_bias_30 = None
        relative_position_bias_31 = permute_31.contiguous()
        permute_31 = None
        rel_pos_bias_15 = relative_position_bias_31.unsqueeze(0)
        relative_position_bias_31 = None
        x_200 = torch._C._nn.scaled_dot_product_attention(
            q_15, k_15, v_15, attn_mask=rel_pos_bias_15, dropout_p=0.0
        )
        q_15 = k_15 = v_15 = rel_pos_bias_15 = None
        transpose_16 = x_200.transpose(1, 2)
        x_200 = None
        x_201 = transpose_16.reshape(1, 197, 1024)
        transpose_16 = None
        x_202 = torch._C._nn.linear(
            x_201,
            l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_,
        )
        x_201 = l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_203 = torch.nn.functional.dropout(x_202, 0.0, False, False)
        x_202 = None
        mul_30 = l_self_modules_blocks_modules_15_parameters_gamma_1_ * x_203
        l_self_modules_blocks_modules_15_parameters_gamma_1_ = x_203 = None
        x_204 = x_198 + mul_30
        x_198 = mul_30 = None
        x_205 = torch.nn.functional.layer_norm(
            x_204,
            (1024,),
            l_self_modules_blocks_modules_15_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_15_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_15_modules_norm2_parameters_bias_
        ) = None
        x_206 = torch._C._nn.linear(
            x_205,
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_205 = (
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_207 = torch._C._nn.gelu(x_206, approximate="none")
        x_206 = None
        x_208 = torch.nn.functional.dropout(x_207, 0.0, False, False)
        x_207 = None
        x_209 = torch._C._nn.linear(
            x_208,
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_208 = (
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_210 = torch.nn.functional.dropout(x_209, 0.0, False, False)
        x_209 = None
        mul_31 = l_self_modules_blocks_modules_15_parameters_gamma_2_ * x_210
        l_self_modules_blocks_modules_15_parameters_gamma_2_ = x_210 = None
        x_211 = x_204 + mul_31
        x_204 = mul_31 = None
        x_212 = torch.nn.functional.layer_norm(
            x_211,
            (1024,),
            l_self_modules_blocks_modules_16_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_16_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_16_modules_norm1_parameters_bias_
        ) = None
        qkv_bias_16 = torch.cat(
            (
                l_self_modules_blocks_modules_16_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_16_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_16_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_16_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_16_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_16_modules_attn_parameters_v_bias_ = None
        qkv_32 = torch._C._nn.linear(
            x_212,
            weight=l_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_16,
        )
        x_212 = (
            l_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_
        ) = qkv_bias_16 = None
        reshape_32 = qkv_32.reshape(1, 197, 3, 16, -1)
        qkv_32 = None
        qkv_33 = reshape_32.permute(2, 0, 3, 1, 4)
        reshape_32 = None
        unbind_16 = qkv_33.unbind(0)
        qkv_33 = None
        q_16 = unbind_16[0]
        k_16 = unbind_16[1]
        v_16 = unbind_16[2]
        unbind_16 = None
        view_32 = l_self_modules_blocks_modules_16_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_blocks_modules_16_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_67 = l_self_modules_blocks_modules_16_modules_attn_parameters_relative_position_bias_table_[
            view_32
        ]
        l_self_modules_blocks_modules_16_modules_attn_parameters_relative_position_bias_table_ = (
            view_32
        ) = None
        relative_position_bias_32 = getitem_67.view(197, 197, -1)
        getitem_67 = None
        permute_33 = relative_position_bias_32.permute(2, 0, 1)
        relative_position_bias_32 = None
        relative_position_bias_33 = permute_33.contiguous()
        permute_33 = None
        rel_pos_bias_16 = relative_position_bias_33.unsqueeze(0)
        relative_position_bias_33 = None
        x_213 = torch._C._nn.scaled_dot_product_attention(
            q_16, k_16, v_16, attn_mask=rel_pos_bias_16, dropout_p=0.0
        )
        q_16 = k_16 = v_16 = rel_pos_bias_16 = None
        transpose_17 = x_213.transpose(1, 2)
        x_213 = None
        x_214 = transpose_17.reshape(1, 197, 1024)
        transpose_17 = None
        x_215 = torch._C._nn.linear(
            x_214,
            l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_,
        )
        x_214 = l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_216 = torch.nn.functional.dropout(x_215, 0.0, False, False)
        x_215 = None
        mul_32 = l_self_modules_blocks_modules_16_parameters_gamma_1_ * x_216
        l_self_modules_blocks_modules_16_parameters_gamma_1_ = x_216 = None
        x_217 = x_211 + mul_32
        x_211 = mul_32 = None
        x_218 = torch.nn.functional.layer_norm(
            x_217,
            (1024,),
            l_self_modules_blocks_modules_16_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_16_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_16_modules_norm2_parameters_bias_
        ) = None
        x_219 = torch._C._nn.linear(
            x_218,
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_218 = (
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_220 = torch._C._nn.gelu(x_219, approximate="none")
        x_219 = None
        x_221 = torch.nn.functional.dropout(x_220, 0.0, False, False)
        x_220 = None
        x_222 = torch._C._nn.linear(
            x_221,
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_221 = (
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_223 = torch.nn.functional.dropout(x_222, 0.0, False, False)
        x_222 = None
        mul_33 = l_self_modules_blocks_modules_16_parameters_gamma_2_ * x_223
        l_self_modules_blocks_modules_16_parameters_gamma_2_ = x_223 = None
        x_224 = x_217 + mul_33
        x_217 = mul_33 = None
        x_225 = torch.nn.functional.layer_norm(
            x_224,
            (1024,),
            l_self_modules_blocks_modules_17_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_17_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_17_modules_norm1_parameters_bias_
        ) = None
        qkv_bias_17 = torch.cat(
            (
                l_self_modules_blocks_modules_17_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_17_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_17_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_17_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_17_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_17_modules_attn_parameters_v_bias_ = None
        qkv_34 = torch._C._nn.linear(
            x_225,
            weight=l_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_17,
        )
        x_225 = (
            l_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_
        ) = qkv_bias_17 = None
        reshape_34 = qkv_34.reshape(1, 197, 3, 16, -1)
        qkv_34 = None
        qkv_35 = reshape_34.permute(2, 0, 3, 1, 4)
        reshape_34 = None
        unbind_17 = qkv_35.unbind(0)
        qkv_35 = None
        q_17 = unbind_17[0]
        k_17 = unbind_17[1]
        v_17 = unbind_17[2]
        unbind_17 = None
        view_34 = l_self_modules_blocks_modules_17_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_blocks_modules_17_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_71 = l_self_modules_blocks_modules_17_modules_attn_parameters_relative_position_bias_table_[
            view_34
        ]
        l_self_modules_blocks_modules_17_modules_attn_parameters_relative_position_bias_table_ = (
            view_34
        ) = None
        relative_position_bias_34 = getitem_71.view(197, 197, -1)
        getitem_71 = None
        permute_35 = relative_position_bias_34.permute(2, 0, 1)
        relative_position_bias_34 = None
        relative_position_bias_35 = permute_35.contiguous()
        permute_35 = None
        rel_pos_bias_17 = relative_position_bias_35.unsqueeze(0)
        relative_position_bias_35 = None
        x_226 = torch._C._nn.scaled_dot_product_attention(
            q_17, k_17, v_17, attn_mask=rel_pos_bias_17, dropout_p=0.0
        )
        q_17 = k_17 = v_17 = rel_pos_bias_17 = None
        transpose_18 = x_226.transpose(1, 2)
        x_226 = None
        x_227 = transpose_18.reshape(1, 197, 1024)
        transpose_18 = None
        x_228 = torch._C._nn.linear(
            x_227,
            l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_,
        )
        x_227 = l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_229 = torch.nn.functional.dropout(x_228, 0.0, False, False)
        x_228 = None
        mul_34 = l_self_modules_blocks_modules_17_parameters_gamma_1_ * x_229
        l_self_modules_blocks_modules_17_parameters_gamma_1_ = x_229 = None
        x_230 = x_224 + mul_34
        x_224 = mul_34 = None
        x_231 = torch.nn.functional.layer_norm(
            x_230,
            (1024,),
            l_self_modules_blocks_modules_17_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_17_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_17_modules_norm2_parameters_bias_
        ) = None
        x_232 = torch._C._nn.linear(
            x_231,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_231 = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_233 = torch._C._nn.gelu(x_232, approximate="none")
        x_232 = None
        x_234 = torch.nn.functional.dropout(x_233, 0.0, False, False)
        x_233 = None
        x_235 = torch._C._nn.linear(
            x_234,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_234 = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_236 = torch.nn.functional.dropout(x_235, 0.0, False, False)
        x_235 = None
        mul_35 = l_self_modules_blocks_modules_17_parameters_gamma_2_ * x_236
        l_self_modules_blocks_modules_17_parameters_gamma_2_ = x_236 = None
        x_237 = x_230 + mul_35
        x_230 = mul_35 = None
        x_238 = torch.nn.functional.layer_norm(
            x_237,
            (1024,),
            l_self_modules_blocks_modules_18_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_18_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_18_modules_norm1_parameters_bias_
        ) = None
        qkv_bias_18 = torch.cat(
            (
                l_self_modules_blocks_modules_18_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_18_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_18_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_18_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_18_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_18_modules_attn_parameters_v_bias_ = None
        qkv_36 = torch._C._nn.linear(
            x_238,
            weight=l_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_18,
        )
        x_238 = (
            l_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_
        ) = qkv_bias_18 = None
        reshape_36 = qkv_36.reshape(1, 197, 3, 16, -1)
        qkv_36 = None
        qkv_37 = reshape_36.permute(2, 0, 3, 1, 4)
        reshape_36 = None
        unbind_18 = qkv_37.unbind(0)
        qkv_37 = None
        q_18 = unbind_18[0]
        k_18 = unbind_18[1]
        v_18 = unbind_18[2]
        unbind_18 = None
        view_36 = l_self_modules_blocks_modules_18_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_blocks_modules_18_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_75 = l_self_modules_blocks_modules_18_modules_attn_parameters_relative_position_bias_table_[
            view_36
        ]
        l_self_modules_blocks_modules_18_modules_attn_parameters_relative_position_bias_table_ = (
            view_36
        ) = None
        relative_position_bias_36 = getitem_75.view(197, 197, -1)
        getitem_75 = None
        permute_37 = relative_position_bias_36.permute(2, 0, 1)
        relative_position_bias_36 = None
        relative_position_bias_37 = permute_37.contiguous()
        permute_37 = None
        rel_pos_bias_18 = relative_position_bias_37.unsqueeze(0)
        relative_position_bias_37 = None
        x_239 = torch._C._nn.scaled_dot_product_attention(
            q_18, k_18, v_18, attn_mask=rel_pos_bias_18, dropout_p=0.0
        )
        q_18 = k_18 = v_18 = rel_pos_bias_18 = None
        transpose_19 = x_239.transpose(1, 2)
        x_239 = None
        x_240 = transpose_19.reshape(1, 197, 1024)
        transpose_19 = None
        x_241 = torch._C._nn.linear(
            x_240,
            l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_,
        )
        x_240 = l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_242 = torch.nn.functional.dropout(x_241, 0.0, False, False)
        x_241 = None
        mul_36 = l_self_modules_blocks_modules_18_parameters_gamma_1_ * x_242
        l_self_modules_blocks_modules_18_parameters_gamma_1_ = x_242 = None
        x_243 = x_237 + mul_36
        x_237 = mul_36 = None
        x_244 = torch.nn.functional.layer_norm(
            x_243,
            (1024,),
            l_self_modules_blocks_modules_18_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_18_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_18_modules_norm2_parameters_bias_
        ) = None
        x_245 = torch._C._nn.linear(
            x_244,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_244 = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_246 = torch._C._nn.gelu(x_245, approximate="none")
        x_245 = None
        x_247 = torch.nn.functional.dropout(x_246, 0.0, False, False)
        x_246 = None
        x_248 = torch._C._nn.linear(
            x_247,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_247 = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_249 = torch.nn.functional.dropout(x_248, 0.0, False, False)
        x_248 = None
        mul_37 = l_self_modules_blocks_modules_18_parameters_gamma_2_ * x_249
        l_self_modules_blocks_modules_18_parameters_gamma_2_ = x_249 = None
        x_250 = x_243 + mul_37
        x_243 = mul_37 = None
        x_251 = torch.nn.functional.layer_norm(
            x_250,
            (1024,),
            l_self_modules_blocks_modules_19_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_19_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_19_modules_norm1_parameters_bias_
        ) = None
        qkv_bias_19 = torch.cat(
            (
                l_self_modules_blocks_modules_19_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_19_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_19_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_19_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_19_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_19_modules_attn_parameters_v_bias_ = None
        qkv_38 = torch._C._nn.linear(
            x_251,
            weight=l_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_19,
        )
        x_251 = (
            l_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_
        ) = qkv_bias_19 = None
        reshape_38 = qkv_38.reshape(1, 197, 3, 16, -1)
        qkv_38 = None
        qkv_39 = reshape_38.permute(2, 0, 3, 1, 4)
        reshape_38 = None
        unbind_19 = qkv_39.unbind(0)
        qkv_39 = None
        q_19 = unbind_19[0]
        k_19 = unbind_19[1]
        v_19 = unbind_19[2]
        unbind_19 = None
        view_38 = l_self_modules_blocks_modules_19_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_blocks_modules_19_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_79 = l_self_modules_blocks_modules_19_modules_attn_parameters_relative_position_bias_table_[
            view_38
        ]
        l_self_modules_blocks_modules_19_modules_attn_parameters_relative_position_bias_table_ = (
            view_38
        ) = None
        relative_position_bias_38 = getitem_79.view(197, 197, -1)
        getitem_79 = None
        permute_39 = relative_position_bias_38.permute(2, 0, 1)
        relative_position_bias_38 = None
        relative_position_bias_39 = permute_39.contiguous()
        permute_39 = None
        rel_pos_bias_19 = relative_position_bias_39.unsqueeze(0)
        relative_position_bias_39 = None
        x_252 = torch._C._nn.scaled_dot_product_attention(
            q_19, k_19, v_19, attn_mask=rel_pos_bias_19, dropout_p=0.0
        )
        q_19 = k_19 = v_19 = rel_pos_bias_19 = None
        transpose_20 = x_252.transpose(1, 2)
        x_252 = None
        x_253 = transpose_20.reshape(1, 197, 1024)
        transpose_20 = None
        x_254 = torch._C._nn.linear(
            x_253,
            l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_,
        )
        x_253 = l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_255 = torch.nn.functional.dropout(x_254, 0.0, False, False)
        x_254 = None
        mul_38 = l_self_modules_blocks_modules_19_parameters_gamma_1_ * x_255
        l_self_modules_blocks_modules_19_parameters_gamma_1_ = x_255 = None
        x_256 = x_250 + mul_38
        x_250 = mul_38 = None
        x_257 = torch.nn.functional.layer_norm(
            x_256,
            (1024,),
            l_self_modules_blocks_modules_19_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_19_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_19_modules_norm2_parameters_bias_
        ) = None
        x_258 = torch._C._nn.linear(
            x_257,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_257 = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_259 = torch._C._nn.gelu(x_258, approximate="none")
        x_258 = None
        x_260 = torch.nn.functional.dropout(x_259, 0.0, False, False)
        x_259 = None
        x_261 = torch._C._nn.linear(
            x_260,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_260 = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_262 = torch.nn.functional.dropout(x_261, 0.0, False, False)
        x_261 = None
        mul_39 = l_self_modules_blocks_modules_19_parameters_gamma_2_ * x_262
        l_self_modules_blocks_modules_19_parameters_gamma_2_ = x_262 = None
        x_263 = x_256 + mul_39
        x_256 = mul_39 = None
        x_264 = torch.nn.functional.layer_norm(
            x_263,
            (1024,),
            l_self_modules_blocks_modules_20_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_20_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_20_modules_norm1_parameters_bias_
        ) = None
        qkv_bias_20 = torch.cat(
            (
                l_self_modules_blocks_modules_20_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_20_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_20_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_20_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_20_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_20_modules_attn_parameters_v_bias_ = None
        qkv_40 = torch._C._nn.linear(
            x_264,
            weight=l_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_20,
        )
        x_264 = (
            l_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_
        ) = qkv_bias_20 = None
        reshape_40 = qkv_40.reshape(1, 197, 3, 16, -1)
        qkv_40 = None
        qkv_41 = reshape_40.permute(2, 0, 3, 1, 4)
        reshape_40 = None
        unbind_20 = qkv_41.unbind(0)
        qkv_41 = None
        q_20 = unbind_20[0]
        k_20 = unbind_20[1]
        v_20 = unbind_20[2]
        unbind_20 = None
        view_40 = l_self_modules_blocks_modules_20_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_blocks_modules_20_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_83 = l_self_modules_blocks_modules_20_modules_attn_parameters_relative_position_bias_table_[
            view_40
        ]
        l_self_modules_blocks_modules_20_modules_attn_parameters_relative_position_bias_table_ = (
            view_40
        ) = None
        relative_position_bias_40 = getitem_83.view(197, 197, -1)
        getitem_83 = None
        permute_41 = relative_position_bias_40.permute(2, 0, 1)
        relative_position_bias_40 = None
        relative_position_bias_41 = permute_41.contiguous()
        permute_41 = None
        rel_pos_bias_20 = relative_position_bias_41.unsqueeze(0)
        relative_position_bias_41 = None
        x_265 = torch._C._nn.scaled_dot_product_attention(
            q_20, k_20, v_20, attn_mask=rel_pos_bias_20, dropout_p=0.0
        )
        q_20 = k_20 = v_20 = rel_pos_bias_20 = None
        transpose_21 = x_265.transpose(1, 2)
        x_265 = None
        x_266 = transpose_21.reshape(1, 197, 1024)
        transpose_21 = None
        x_267 = torch._C._nn.linear(
            x_266,
            l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_,
        )
        x_266 = l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_268 = torch.nn.functional.dropout(x_267, 0.0, False, False)
        x_267 = None
        mul_40 = l_self_modules_blocks_modules_20_parameters_gamma_1_ * x_268
        l_self_modules_blocks_modules_20_parameters_gamma_1_ = x_268 = None
        x_269 = x_263 + mul_40
        x_263 = mul_40 = None
        x_270 = torch.nn.functional.layer_norm(
            x_269,
            (1024,),
            l_self_modules_blocks_modules_20_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_20_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_20_modules_norm2_parameters_bias_
        ) = None
        x_271 = torch._C._nn.linear(
            x_270,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_270 = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_272 = torch._C._nn.gelu(x_271, approximate="none")
        x_271 = None
        x_273 = torch.nn.functional.dropout(x_272, 0.0, False, False)
        x_272 = None
        x_274 = torch._C._nn.linear(
            x_273,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_273 = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_275 = torch.nn.functional.dropout(x_274, 0.0, False, False)
        x_274 = None
        mul_41 = l_self_modules_blocks_modules_20_parameters_gamma_2_ * x_275
        l_self_modules_blocks_modules_20_parameters_gamma_2_ = x_275 = None
        x_276 = x_269 + mul_41
        x_269 = mul_41 = None
        x_277 = torch.nn.functional.layer_norm(
            x_276,
            (1024,),
            l_self_modules_blocks_modules_21_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_21_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_21_modules_norm1_parameters_bias_
        ) = None
        qkv_bias_21 = torch.cat(
            (
                l_self_modules_blocks_modules_21_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_21_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_21_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_21_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_21_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_21_modules_attn_parameters_v_bias_ = None
        qkv_42 = torch._C._nn.linear(
            x_277,
            weight=l_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_21,
        )
        x_277 = (
            l_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_
        ) = qkv_bias_21 = None
        reshape_42 = qkv_42.reshape(1, 197, 3, 16, -1)
        qkv_42 = None
        qkv_43 = reshape_42.permute(2, 0, 3, 1, 4)
        reshape_42 = None
        unbind_21 = qkv_43.unbind(0)
        qkv_43 = None
        q_21 = unbind_21[0]
        k_21 = unbind_21[1]
        v_21 = unbind_21[2]
        unbind_21 = None
        view_42 = l_self_modules_blocks_modules_21_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_blocks_modules_21_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_87 = l_self_modules_blocks_modules_21_modules_attn_parameters_relative_position_bias_table_[
            view_42
        ]
        l_self_modules_blocks_modules_21_modules_attn_parameters_relative_position_bias_table_ = (
            view_42
        ) = None
        relative_position_bias_42 = getitem_87.view(197, 197, -1)
        getitem_87 = None
        permute_43 = relative_position_bias_42.permute(2, 0, 1)
        relative_position_bias_42 = None
        relative_position_bias_43 = permute_43.contiguous()
        permute_43 = None
        rel_pos_bias_21 = relative_position_bias_43.unsqueeze(0)
        relative_position_bias_43 = None
        x_278 = torch._C._nn.scaled_dot_product_attention(
            q_21, k_21, v_21, attn_mask=rel_pos_bias_21, dropout_p=0.0
        )
        q_21 = k_21 = v_21 = rel_pos_bias_21 = None
        transpose_22 = x_278.transpose(1, 2)
        x_278 = None
        x_279 = transpose_22.reshape(1, 197, 1024)
        transpose_22 = None
        x_280 = torch._C._nn.linear(
            x_279,
            l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_,
        )
        x_279 = l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_281 = torch.nn.functional.dropout(x_280, 0.0, False, False)
        x_280 = None
        mul_42 = l_self_modules_blocks_modules_21_parameters_gamma_1_ * x_281
        l_self_modules_blocks_modules_21_parameters_gamma_1_ = x_281 = None
        x_282 = x_276 + mul_42
        x_276 = mul_42 = None
        x_283 = torch.nn.functional.layer_norm(
            x_282,
            (1024,),
            l_self_modules_blocks_modules_21_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_21_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_21_modules_norm2_parameters_bias_
        ) = None
        x_284 = torch._C._nn.linear(
            x_283,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_283 = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_285 = torch._C._nn.gelu(x_284, approximate="none")
        x_284 = None
        x_286 = torch.nn.functional.dropout(x_285, 0.0, False, False)
        x_285 = None
        x_287 = torch._C._nn.linear(
            x_286,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_286 = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_288 = torch.nn.functional.dropout(x_287, 0.0, False, False)
        x_287 = None
        mul_43 = l_self_modules_blocks_modules_21_parameters_gamma_2_ * x_288
        l_self_modules_blocks_modules_21_parameters_gamma_2_ = x_288 = None
        x_289 = x_282 + mul_43
        x_282 = mul_43 = None
        x_290 = torch.nn.functional.layer_norm(
            x_289,
            (1024,),
            l_self_modules_blocks_modules_22_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_22_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_22_modules_norm1_parameters_bias_
        ) = None
        qkv_bias_22 = torch.cat(
            (
                l_self_modules_blocks_modules_22_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_22_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_22_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_22_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_22_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_22_modules_attn_parameters_v_bias_ = None
        qkv_44 = torch._C._nn.linear(
            x_290,
            weight=l_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_22,
        )
        x_290 = (
            l_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_
        ) = qkv_bias_22 = None
        reshape_44 = qkv_44.reshape(1, 197, 3, 16, -1)
        qkv_44 = None
        qkv_45 = reshape_44.permute(2, 0, 3, 1, 4)
        reshape_44 = None
        unbind_22 = qkv_45.unbind(0)
        qkv_45 = None
        q_22 = unbind_22[0]
        k_22 = unbind_22[1]
        v_22 = unbind_22[2]
        unbind_22 = None
        view_44 = l_self_modules_blocks_modules_22_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_blocks_modules_22_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_91 = l_self_modules_blocks_modules_22_modules_attn_parameters_relative_position_bias_table_[
            view_44
        ]
        l_self_modules_blocks_modules_22_modules_attn_parameters_relative_position_bias_table_ = (
            view_44
        ) = None
        relative_position_bias_44 = getitem_91.view(197, 197, -1)
        getitem_91 = None
        permute_45 = relative_position_bias_44.permute(2, 0, 1)
        relative_position_bias_44 = None
        relative_position_bias_45 = permute_45.contiguous()
        permute_45 = None
        rel_pos_bias_22 = relative_position_bias_45.unsqueeze(0)
        relative_position_bias_45 = None
        x_291 = torch._C._nn.scaled_dot_product_attention(
            q_22, k_22, v_22, attn_mask=rel_pos_bias_22, dropout_p=0.0
        )
        q_22 = k_22 = v_22 = rel_pos_bias_22 = None
        transpose_23 = x_291.transpose(1, 2)
        x_291 = None
        x_292 = transpose_23.reshape(1, 197, 1024)
        transpose_23 = None
        x_293 = torch._C._nn.linear(
            x_292,
            l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_,
        )
        x_292 = l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_294 = torch.nn.functional.dropout(x_293, 0.0, False, False)
        x_293 = None
        mul_44 = l_self_modules_blocks_modules_22_parameters_gamma_1_ * x_294
        l_self_modules_blocks_modules_22_parameters_gamma_1_ = x_294 = None
        x_295 = x_289 + mul_44
        x_289 = mul_44 = None
        x_296 = torch.nn.functional.layer_norm(
            x_295,
            (1024,),
            l_self_modules_blocks_modules_22_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_22_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_22_modules_norm2_parameters_bias_
        ) = None
        x_297 = torch._C._nn.linear(
            x_296,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_296 = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_298 = torch._C._nn.gelu(x_297, approximate="none")
        x_297 = None
        x_299 = torch.nn.functional.dropout(x_298, 0.0, False, False)
        x_298 = None
        x_300 = torch._C._nn.linear(
            x_299,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_299 = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_301 = torch.nn.functional.dropout(x_300, 0.0, False, False)
        x_300 = None
        mul_45 = l_self_modules_blocks_modules_22_parameters_gamma_2_ * x_301
        l_self_modules_blocks_modules_22_parameters_gamma_2_ = x_301 = None
        x_302 = x_295 + mul_45
        x_295 = mul_45 = None
        x_303 = torch.nn.functional.layer_norm(
            x_302,
            (1024,),
            l_self_modules_blocks_modules_23_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_23_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_23_modules_norm1_parameters_bias_
        ) = None
        qkv_bias_23 = torch.cat(
            (
                l_self_modules_blocks_modules_23_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_23_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_23_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_23_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_23_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_23_modules_attn_parameters_v_bias_ = None
        qkv_46 = torch._C._nn.linear(
            x_303,
            weight=l_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_23,
        )
        x_303 = (
            l_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_
        ) = qkv_bias_23 = None
        reshape_46 = qkv_46.reshape(1, 197, 3, 16, -1)
        qkv_46 = None
        qkv_47 = reshape_46.permute(2, 0, 3, 1, 4)
        reshape_46 = None
        unbind_23 = qkv_47.unbind(0)
        qkv_47 = None
        q_23 = unbind_23[0]
        k_23 = unbind_23[1]
        v_23 = unbind_23[2]
        unbind_23 = None
        view_46 = l_self_modules_blocks_modules_23_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_blocks_modules_23_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_95 = l_self_modules_blocks_modules_23_modules_attn_parameters_relative_position_bias_table_[
            view_46
        ]
        l_self_modules_blocks_modules_23_modules_attn_parameters_relative_position_bias_table_ = (
            view_46
        ) = None
        relative_position_bias_46 = getitem_95.view(197, 197, -1)
        getitem_95 = None
        permute_47 = relative_position_bias_46.permute(2, 0, 1)
        relative_position_bias_46 = None
        relative_position_bias_47 = permute_47.contiguous()
        permute_47 = None
        rel_pos_bias_23 = relative_position_bias_47.unsqueeze(0)
        relative_position_bias_47 = None
        x_304 = torch._C._nn.scaled_dot_product_attention(
            q_23, k_23, v_23, attn_mask=rel_pos_bias_23, dropout_p=0.0
        )
        q_23 = k_23 = v_23 = rel_pos_bias_23 = None
        transpose_24 = x_304.transpose(1, 2)
        x_304 = None
        x_305 = transpose_24.reshape(1, 197, 1024)
        transpose_24 = None
        x_306 = torch._C._nn.linear(
            x_305,
            l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_,
        )
        x_305 = l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_307 = torch.nn.functional.dropout(x_306, 0.0, False, False)
        x_306 = None
        mul_46 = l_self_modules_blocks_modules_23_parameters_gamma_1_ * x_307
        l_self_modules_blocks_modules_23_parameters_gamma_1_ = x_307 = None
        x_308 = x_302 + mul_46
        x_302 = mul_46 = None
        x_309 = torch.nn.functional.layer_norm(
            x_308,
            (1024,),
            l_self_modules_blocks_modules_23_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_23_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_23_modules_norm2_parameters_bias_
        ) = None
        x_310 = torch._C._nn.linear(
            x_309,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_309 = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_311 = torch._C._nn.gelu(x_310, approximate="none")
        x_310 = None
        x_312 = torch.nn.functional.dropout(x_311, 0.0, False, False)
        x_311 = None
        x_313 = torch._C._nn.linear(
            x_312,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_312 = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_314 = torch.nn.functional.dropout(x_313, 0.0, False, False)
        x_313 = None
        mul_47 = l_self_modules_blocks_modules_23_parameters_gamma_2_ * x_314
        l_self_modules_blocks_modules_23_parameters_gamma_2_ = x_314 = None
        x_315 = x_308 + mul_47
        x_308 = mul_47 = None
        getitem_96 = x_315[(slice(None, None, None), slice(1, None, None))]
        x_315 = None
        x_316 = getitem_96.mean(dim=1)
        getitem_96 = None
        x_317 = torch.nn.functional.layer_norm(
            x_316,
            (1024,),
            l_self_modules_fc_norm_parameters_weight_,
            l_self_modules_fc_norm_parameters_bias_,
            1e-06,
        )
        x_316 = (
            l_self_modules_fc_norm_parameters_weight_
        ) = l_self_modules_fc_norm_parameters_bias_ = None
        x_318 = torch.nn.functional.dropout(x_317, 0.0, False, False)
        x_317 = None
        x_319 = torch._C._nn.linear(
            x_318,
            l_self_modules_head_parameters_weight_,
            l_self_modules_head_parameters_bias_,
        )
        x_318 = (
            l_self_modules_head_parameters_weight_
        ) = l_self_modules_head_parameters_bias_ = None
        return (x_319,)
