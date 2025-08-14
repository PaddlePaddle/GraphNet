import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_features_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_x_: torch.Tensor,
        L_self_modules_features_modules_0_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_0_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_reduction_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_3_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_3_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_reduction_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_5_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_5_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_5_modules_2_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_5_modules_3_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_5_modules_4_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_5_modules_5_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_5_modules_6_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_5_modules_7_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_5_modules_8_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_5_modules_9_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_5_modules_10_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_5_modules_11_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_5_modules_12_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_5_modules_13_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_5_modules_14_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_5_modules_15_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_5_modules_16_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_5_modules_17_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_reduction_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_7_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_7_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_features_modules_0_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_0_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_0_modules_0_parameters_bias_ = (
            L_self_modules_features_modules_0_modules_0_parameters_bias_
        )
        l_x_ = L_x_
        l_self_modules_features_modules_0_modules_2_parameters_weight_ = (
            L_self_modules_features_modules_0_modules_2_parameters_weight_
        )
        l_self_modules_features_modules_0_modules_2_parameters_bias_ = (
            L_self_modules_features_modules_0_modules_2_parameters_bias_
        )
        l_self_modules_features_modules_1_modules_0_modules_norm1_parameters_weight_ = (
            L_self_modules_features_modules_1_modules_0_modules_norm1_parameters_weight_
        )
        l_self_modules_features_modules_1_modules_0_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_1_modules_0_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_1_modules_0_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_features_modules_1_modules_0_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_features_modules_1_modules_0_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_1_modules_0_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_1_modules_0_modules_norm2_parameters_weight_ = (
            L_self_modules_features_modules_1_modules_0_modules_norm2_parameters_weight_
        )
        l_self_modules_features_modules_1_modules_0_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_1_modules_0_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_1_modules_0_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_1_modules_0_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_1_modules_0_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_1_modules_0_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_1_modules_0_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_1_modules_0_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_1_modules_0_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_1_modules_0_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_1_modules_1_modules_norm1_parameters_weight_ = (
            L_self_modules_features_modules_1_modules_1_modules_norm1_parameters_weight_
        )
        l_self_modules_features_modules_1_modules_1_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_1_modules_1_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_1_modules_1_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_features_modules_1_modules_1_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_features_modules_1_modules_1_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_1_modules_1_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_1_modules_1_modules_norm2_parameters_weight_ = (
            L_self_modules_features_modules_1_modules_1_modules_norm2_parameters_weight_
        )
        l_self_modules_features_modules_1_modules_1_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_1_modules_1_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_1_modules_1_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_1_modules_1_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_1_modules_1_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_1_modules_1_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_1_modules_1_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_1_modules_1_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_1_modules_1_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_1_modules_1_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_2_modules_norm_parameters_weight_ = (
            L_self_modules_features_modules_2_modules_norm_parameters_weight_
        )
        l_self_modules_features_modules_2_modules_norm_parameters_bias_ = (
            L_self_modules_features_modules_2_modules_norm_parameters_bias_
        )
        l_self_modules_features_modules_2_modules_reduction_parameters_weight_ = (
            L_self_modules_features_modules_2_modules_reduction_parameters_weight_
        )
        l_self_modules_features_modules_3_modules_0_modules_norm1_parameters_weight_ = (
            L_self_modules_features_modules_3_modules_0_modules_norm1_parameters_weight_
        )
        l_self_modules_features_modules_3_modules_0_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_3_modules_0_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_3_modules_0_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_features_modules_3_modules_0_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_features_modules_3_modules_0_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_3_modules_0_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_3_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_3_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_3_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_3_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_3_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_3_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_3_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_3_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_3_modules_0_modules_norm2_parameters_weight_ = (
            L_self_modules_features_modules_3_modules_0_modules_norm2_parameters_weight_
        )
        l_self_modules_features_modules_3_modules_0_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_3_modules_0_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_3_modules_0_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_3_modules_0_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_3_modules_0_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_3_modules_0_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_3_modules_0_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_3_modules_0_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_3_modules_0_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_3_modules_0_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_3_modules_1_modules_norm1_parameters_weight_ = (
            L_self_modules_features_modules_3_modules_1_modules_norm1_parameters_weight_
        )
        l_self_modules_features_modules_3_modules_1_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_3_modules_1_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_3_modules_1_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_features_modules_3_modules_1_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_features_modules_3_modules_1_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_3_modules_1_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_3_modules_1_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_3_modules_1_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_3_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_3_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_3_modules_1_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_3_modules_1_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_3_modules_1_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_3_modules_1_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_3_modules_1_modules_norm2_parameters_weight_ = (
            L_self_modules_features_modules_3_modules_1_modules_norm2_parameters_weight_
        )
        l_self_modules_features_modules_3_modules_1_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_3_modules_1_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_3_modules_1_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_3_modules_1_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_3_modules_1_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_3_modules_1_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_3_modules_1_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_3_modules_1_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_3_modules_1_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_3_modules_1_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_4_modules_norm_parameters_weight_ = (
            L_self_modules_features_modules_4_modules_norm_parameters_weight_
        )
        l_self_modules_features_modules_4_modules_norm_parameters_bias_ = (
            L_self_modules_features_modules_4_modules_norm_parameters_bias_
        )
        l_self_modules_features_modules_4_modules_reduction_parameters_weight_ = (
            L_self_modules_features_modules_4_modules_reduction_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_0_modules_norm1_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_0_modules_norm1_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_0_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_0_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_0_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_features_modules_5_modules_0_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_features_modules_5_modules_0_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_5_modules_0_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_5_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_5_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_5_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_5_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_5_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_5_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_5_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_5_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_5_modules_0_modules_norm2_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_0_modules_norm2_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_0_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_0_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_0_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_0_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_0_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_0_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_0_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_0_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_0_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_0_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_1_modules_norm1_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_1_modules_norm1_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_1_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_1_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_1_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_features_modules_5_modules_1_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_features_modules_5_modules_1_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_5_modules_1_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_5_modules_1_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_5_modules_1_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_5_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_5_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_5_modules_1_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_5_modules_1_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_5_modules_1_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_5_modules_1_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_5_modules_1_modules_norm2_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_1_modules_norm2_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_1_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_1_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_1_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_1_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_1_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_1_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_1_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_1_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_1_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_1_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_2_modules_norm1_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_2_modules_norm1_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_2_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_2_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_2_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_features_modules_5_modules_2_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_features_modules_5_modules_2_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_5_modules_2_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_5_modules_2_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_5_modules_2_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_5_modules_2_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_5_modules_2_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_5_modules_2_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_5_modules_2_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_5_modules_2_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_5_modules_2_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_5_modules_2_modules_norm2_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_2_modules_norm2_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_2_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_2_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_2_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_2_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_2_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_2_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_2_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_2_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_2_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_2_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_3_modules_norm1_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_3_modules_norm1_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_3_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_3_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_3_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_features_modules_5_modules_3_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_features_modules_5_modules_3_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_5_modules_3_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_5_modules_3_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_5_modules_3_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_5_modules_3_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_5_modules_3_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_5_modules_3_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_5_modules_3_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_5_modules_3_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_5_modules_3_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_5_modules_3_modules_norm2_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_3_modules_norm2_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_3_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_3_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_3_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_3_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_3_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_3_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_3_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_3_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_3_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_3_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_4_modules_norm1_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_4_modules_norm1_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_4_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_4_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_4_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_features_modules_5_modules_4_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_features_modules_5_modules_4_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_5_modules_4_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_5_modules_4_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_5_modules_4_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_5_modules_4_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_5_modules_4_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_5_modules_4_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_5_modules_4_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_5_modules_4_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_5_modules_4_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_5_modules_4_modules_norm2_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_4_modules_norm2_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_4_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_4_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_4_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_4_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_4_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_4_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_4_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_4_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_4_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_4_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_5_modules_norm1_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_5_modules_norm1_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_5_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_5_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_5_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_features_modules_5_modules_5_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_features_modules_5_modules_5_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_5_modules_5_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_5_modules_5_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_5_modules_5_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_5_modules_5_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_5_modules_5_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_5_modules_5_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_5_modules_5_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_5_modules_5_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_5_modules_5_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_5_modules_5_modules_norm2_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_5_modules_norm2_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_5_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_5_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_5_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_5_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_5_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_5_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_5_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_5_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_5_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_5_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_6_modules_norm1_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_6_modules_norm1_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_6_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_6_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_6_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_features_modules_5_modules_6_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_features_modules_5_modules_6_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_5_modules_6_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_5_modules_6_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_5_modules_6_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_5_modules_6_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_5_modules_6_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_5_modules_6_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_5_modules_6_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_5_modules_6_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_5_modules_6_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_5_modules_6_modules_norm2_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_6_modules_norm2_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_6_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_6_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_6_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_6_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_6_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_6_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_6_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_6_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_6_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_6_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_7_modules_norm1_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_7_modules_norm1_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_7_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_7_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_7_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_features_modules_5_modules_7_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_features_modules_5_modules_7_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_5_modules_7_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_5_modules_7_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_5_modules_7_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_5_modules_7_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_5_modules_7_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_5_modules_7_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_5_modules_7_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_5_modules_7_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_5_modules_7_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_5_modules_7_modules_norm2_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_7_modules_norm2_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_7_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_7_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_7_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_7_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_7_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_7_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_7_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_7_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_7_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_7_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_8_modules_norm1_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_8_modules_norm1_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_8_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_8_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_8_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_features_modules_5_modules_8_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_features_modules_5_modules_8_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_5_modules_8_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_5_modules_8_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_5_modules_8_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_5_modules_8_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_5_modules_8_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_5_modules_8_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_5_modules_8_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_5_modules_8_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_5_modules_8_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_5_modules_8_modules_norm2_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_8_modules_norm2_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_8_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_8_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_8_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_8_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_8_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_8_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_8_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_8_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_8_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_8_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_9_modules_norm1_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_9_modules_norm1_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_9_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_9_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_9_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_features_modules_5_modules_9_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_features_modules_5_modules_9_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_5_modules_9_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_5_modules_9_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_5_modules_9_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_5_modules_9_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_5_modules_9_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_5_modules_9_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_5_modules_9_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_5_modules_9_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_5_modules_9_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_5_modules_9_modules_norm2_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_9_modules_norm2_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_9_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_9_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_9_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_9_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_9_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_9_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_9_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_9_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_9_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_9_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_10_modules_norm1_parameters_weight_ = L_self_modules_features_modules_5_modules_10_modules_norm1_parameters_weight_
        l_self_modules_features_modules_5_modules_10_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_10_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_10_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_features_modules_5_modules_10_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_features_modules_5_modules_10_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_5_modules_10_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_5_modules_10_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_5_modules_10_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_5_modules_10_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_5_modules_10_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_5_modules_10_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_5_modules_10_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_5_modules_10_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_5_modules_10_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_5_modules_10_modules_norm2_parameters_weight_ = L_self_modules_features_modules_5_modules_10_modules_norm2_parameters_weight_
        l_self_modules_features_modules_5_modules_10_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_10_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_10_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_10_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_10_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_10_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_10_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_10_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_10_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_10_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_11_modules_norm1_parameters_weight_ = L_self_modules_features_modules_5_modules_11_modules_norm1_parameters_weight_
        l_self_modules_features_modules_5_modules_11_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_11_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_11_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_features_modules_5_modules_11_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_features_modules_5_modules_11_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_5_modules_11_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_5_modules_11_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_5_modules_11_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_5_modules_11_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_5_modules_11_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_5_modules_11_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_5_modules_11_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_5_modules_11_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_5_modules_11_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_5_modules_11_modules_norm2_parameters_weight_ = L_self_modules_features_modules_5_modules_11_modules_norm2_parameters_weight_
        l_self_modules_features_modules_5_modules_11_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_11_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_11_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_11_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_11_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_11_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_11_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_11_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_11_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_11_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_12_modules_norm1_parameters_weight_ = L_self_modules_features_modules_5_modules_12_modules_norm1_parameters_weight_
        l_self_modules_features_modules_5_modules_12_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_12_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_12_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_features_modules_5_modules_12_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_features_modules_5_modules_12_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_5_modules_12_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_5_modules_12_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_5_modules_12_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_5_modules_12_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_5_modules_12_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_5_modules_12_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_5_modules_12_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_5_modules_12_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_5_modules_12_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_5_modules_12_modules_norm2_parameters_weight_ = L_self_modules_features_modules_5_modules_12_modules_norm2_parameters_weight_
        l_self_modules_features_modules_5_modules_12_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_12_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_12_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_12_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_12_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_12_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_12_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_12_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_12_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_12_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_13_modules_norm1_parameters_weight_ = L_self_modules_features_modules_5_modules_13_modules_norm1_parameters_weight_
        l_self_modules_features_modules_5_modules_13_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_13_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_13_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_features_modules_5_modules_13_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_features_modules_5_modules_13_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_5_modules_13_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_5_modules_13_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_5_modules_13_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_5_modules_13_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_5_modules_13_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_5_modules_13_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_5_modules_13_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_5_modules_13_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_5_modules_13_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_5_modules_13_modules_norm2_parameters_weight_ = L_self_modules_features_modules_5_modules_13_modules_norm2_parameters_weight_
        l_self_modules_features_modules_5_modules_13_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_13_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_13_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_13_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_13_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_13_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_13_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_13_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_13_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_13_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_14_modules_norm1_parameters_weight_ = L_self_modules_features_modules_5_modules_14_modules_norm1_parameters_weight_
        l_self_modules_features_modules_5_modules_14_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_14_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_14_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_features_modules_5_modules_14_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_features_modules_5_modules_14_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_5_modules_14_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_5_modules_14_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_5_modules_14_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_5_modules_14_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_5_modules_14_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_5_modules_14_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_5_modules_14_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_5_modules_14_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_5_modules_14_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_5_modules_14_modules_norm2_parameters_weight_ = L_self_modules_features_modules_5_modules_14_modules_norm2_parameters_weight_
        l_self_modules_features_modules_5_modules_14_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_14_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_14_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_14_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_14_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_14_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_14_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_14_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_14_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_14_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_15_modules_norm1_parameters_weight_ = L_self_modules_features_modules_5_modules_15_modules_norm1_parameters_weight_
        l_self_modules_features_modules_5_modules_15_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_15_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_15_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_features_modules_5_modules_15_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_features_modules_5_modules_15_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_5_modules_15_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_5_modules_15_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_5_modules_15_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_5_modules_15_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_5_modules_15_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_5_modules_15_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_5_modules_15_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_5_modules_15_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_5_modules_15_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_5_modules_15_modules_norm2_parameters_weight_ = L_self_modules_features_modules_5_modules_15_modules_norm2_parameters_weight_
        l_self_modules_features_modules_5_modules_15_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_15_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_15_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_15_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_15_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_15_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_15_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_15_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_15_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_15_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_16_modules_norm1_parameters_weight_ = L_self_modules_features_modules_5_modules_16_modules_norm1_parameters_weight_
        l_self_modules_features_modules_5_modules_16_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_16_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_16_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_features_modules_5_modules_16_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_features_modules_5_modules_16_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_5_modules_16_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_5_modules_16_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_5_modules_16_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_5_modules_16_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_5_modules_16_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_5_modules_16_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_5_modules_16_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_5_modules_16_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_5_modules_16_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_5_modules_16_modules_norm2_parameters_weight_ = L_self_modules_features_modules_5_modules_16_modules_norm2_parameters_weight_
        l_self_modules_features_modules_5_modules_16_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_16_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_16_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_16_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_16_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_16_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_16_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_16_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_16_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_16_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_17_modules_norm1_parameters_weight_ = L_self_modules_features_modules_5_modules_17_modules_norm1_parameters_weight_
        l_self_modules_features_modules_5_modules_17_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_17_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_17_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_features_modules_5_modules_17_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_features_modules_5_modules_17_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_5_modules_17_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_5_modules_17_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_5_modules_17_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_5_modules_17_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_5_modules_17_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_5_modules_17_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_5_modules_17_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_5_modules_17_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_5_modules_17_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_5_modules_17_modules_norm2_parameters_weight_ = L_self_modules_features_modules_5_modules_17_modules_norm2_parameters_weight_
        l_self_modules_features_modules_5_modules_17_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_17_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_17_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_17_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_17_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_17_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_17_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_17_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_17_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_17_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_6_modules_norm_parameters_weight_ = (
            L_self_modules_features_modules_6_modules_norm_parameters_weight_
        )
        l_self_modules_features_modules_6_modules_norm_parameters_bias_ = (
            L_self_modules_features_modules_6_modules_norm_parameters_bias_
        )
        l_self_modules_features_modules_6_modules_reduction_parameters_weight_ = (
            L_self_modules_features_modules_6_modules_reduction_parameters_weight_
        )
        l_self_modules_features_modules_7_modules_0_modules_norm1_parameters_weight_ = (
            L_self_modules_features_modules_7_modules_0_modules_norm1_parameters_weight_
        )
        l_self_modules_features_modules_7_modules_0_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_7_modules_0_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_7_modules_0_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_features_modules_7_modules_0_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_features_modules_7_modules_0_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_7_modules_0_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_7_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_7_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_7_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_7_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_7_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_7_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_7_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_7_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_7_modules_0_modules_norm2_parameters_weight_ = (
            L_self_modules_features_modules_7_modules_0_modules_norm2_parameters_weight_
        )
        l_self_modules_features_modules_7_modules_0_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_7_modules_0_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_7_modules_0_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_7_modules_0_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_7_modules_0_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_7_modules_0_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_7_modules_0_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_7_modules_0_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_7_modules_0_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_7_modules_0_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_7_modules_1_modules_norm1_parameters_weight_ = (
            L_self_modules_features_modules_7_modules_1_modules_norm1_parameters_weight_
        )
        l_self_modules_features_modules_7_modules_1_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_7_modules_1_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_7_modules_1_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_features_modules_7_modules_1_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_features_modules_7_modules_1_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_7_modules_1_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_7_modules_1_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_7_modules_1_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_7_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_7_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_7_modules_1_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_7_modules_1_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_7_modules_1_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_7_modules_1_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_7_modules_1_modules_norm2_parameters_weight_ = (
            L_self_modules_features_modules_7_modules_1_modules_norm2_parameters_weight_
        )
        l_self_modules_features_modules_7_modules_1_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_7_modules_1_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_7_modules_1_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_7_modules_1_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_7_modules_1_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_7_modules_1_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_7_modules_1_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_7_modules_1_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_7_modules_1_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_7_modules_1_modules_mlp_modules_3_parameters_bias_
        l_self_modules_norm_parameters_weight_ = L_self_modules_norm_parameters_weight_
        l_self_modules_norm_parameters_bias_ = L_self_modules_norm_parameters_bias_
        l_self_modules_head_parameters_weight_ = L_self_modules_head_parameters_weight_
        l_self_modules_head_parameters_bias_ = L_self_modules_head_parameters_bias_
        input_1 = torch.conv2d(
            l_x_,
            l_self_modules_features_modules_0_modules_0_parameters_weight_,
            l_self_modules_features_modules_0_modules_0_parameters_bias_,
            (4, 4),
            (0, 0),
            (1, 1),
            1,
        )
        l_x_ = (
            l_self_modules_features_modules_0_modules_0_parameters_weight_
        ) = l_self_modules_features_modules_0_modules_0_parameters_bias_ = None
        input_2 = torch.permute(input_1, [0, 2, 3, 1])
        input_1 = None
        input_3 = torch.nn.functional.layer_norm(
            input_2,
            (128,),
            l_self_modules_features_modules_0_modules_2_parameters_weight_,
            l_self_modules_features_modules_0_modules_2_parameters_bias_,
            1e-05,
        )
        input_2 = (
            l_self_modules_features_modules_0_modules_2_parameters_weight_
        ) = l_self_modules_features_modules_0_modules_2_parameters_bias_ = None
        layer_norm_1 = torch.nn.functional.layer_norm(
            input_3,
            (128,),
            l_self_modules_features_modules_1_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_1_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_1_modules_0_modules_norm1_parameters_weight_ = (
            l_self_modules_features_modules_1_modules_0_modules_norm1_parameters_bias_
        ) = None
        relative_position_bias = l_self_modules_features_modules_1_modules_0_modules_attn_parameters_relative_position_bias_table_[
            l_self_modules_features_modules_1_modules_0_modules_attn_buffers_relative_position_index_
        ]
        l_self_modules_features_modules_1_modules_0_modules_attn_parameters_relative_position_bias_table_ = l_self_modules_features_modules_1_modules_0_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_1 = relative_position_bias.view(49, 49, -1)
        relative_position_bias = None
        permute_1 = relative_position_bias_1.permute(2, 0, 1)
        relative_position_bias_1 = None
        contiguous = permute_1.contiguous()
        permute_1 = None
        relative_position_bias_2 = contiguous.unsqueeze(0)
        contiguous = None
        x = torch._C._nn.pad(layer_norm_1, (0, 0, 0, 0, 0, 0), "constant", None)
        layer_norm_1 = None
        x_1 = x.view(1, 8, 7, 8, 7, 128)
        x = None
        permute_2 = x_1.permute(0, 1, 3, 2, 4, 5)
        x_1 = None
        x_2 = permute_2.reshape(64, 49, 128)
        permute_2 = None
        qkv = torch._C._nn.linear(
            x_2,
            l_self_modules_features_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_features_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        x_2 = l_self_modules_features_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_features_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_1 = qkv.reshape(64, 49, 3, 4, 32)
        qkv = None
        qkv_1 = reshape_1.permute(2, 0, 3, 1, 4)
        reshape_1 = None
        q = qkv_1[0]
        k = qkv_1[1]
        v = qkv_1[2]
        qkv_1 = None
        q_1 = q * 0.1767766952966369
        q = None
        transpose = k.transpose(-2, -1)
        k = None
        attn = q_1.matmul(transpose)
        q_1 = transpose = None
        attn_1 = attn + relative_position_bias_2
        attn = relative_position_bias_2 = None
        attn_2 = torch.nn.functional.softmax(attn_1, dim=-1)
        attn_1 = None
        attn_3 = torch.nn.functional.dropout(attn_2, p=0.0, training=False)
        attn_2 = None
        matmul_1 = attn_3.matmul(v)
        attn_3 = v = None
        transpose_1 = matmul_1.transpose(1, 2)
        matmul_1 = None
        x_3 = transpose_1.reshape(64, 49, 128)
        transpose_1 = None
        x_4 = torch._C._nn.linear(
            x_3,
            l_self_modules_features_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_3 = l_self_modules_features_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_5 = torch.nn.functional.dropout(x_4, p=0.0, training=False)
        x_4 = None
        x_6 = x_5.view(1, 8, 8, 7, 7, 128)
        x_5 = None
        permute_4 = x_6.permute(0, 1, 3, 2, 4, 5)
        x_6 = None
        x_7 = permute_4.reshape(1, 56, 56, 128)
        permute_4 = None
        getitem_4 = x_7[
            (
                slice(None, None, None),
                slice(None, 56, None),
                slice(None, 56, None),
                slice(None, None, None),
            )
        ]
        x_7 = None
        x_8 = getitem_4.contiguous()
        getitem_4 = None
        _log_api_usage_once = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once = None
        x_9 = input_3 + x_8
        input_3 = x_8 = None
        layer_norm_2 = torch.nn.functional.layer_norm(
            x_9,
            (128,),
            l_self_modules_features_modules_1_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_1_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_1_modules_0_modules_norm2_parameters_weight_ = (
            l_self_modules_features_modules_1_modules_0_modules_norm2_parameters_bias_
        ) = None
        input_4 = torch._C._nn.linear(
            layer_norm_2,
            l_self_modules_features_modules_1_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_1_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        layer_norm_2 = l_self_modules_features_modules_1_modules_0_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_1_modules_0_modules_mlp_modules_0_parameters_bias_ = (None)
        input_5 = torch._C._nn.gelu(input_4, approximate="none")
        input_4 = None
        input_6 = torch.nn.functional.dropout(input_5, 0.0, False, False)
        input_5 = None
        input_7 = torch._C._nn.linear(
            input_6,
            l_self_modules_features_modules_1_modules_0_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_1_modules_0_modules_mlp_modules_3_parameters_bias_,
        )
        input_6 = l_self_modules_features_modules_1_modules_0_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_1_modules_0_modules_mlp_modules_3_parameters_bias_ = (None)
        input_8 = torch.nn.functional.dropout(input_7, 0.0, False, False)
        input_7 = None
        _log_api_usage_once_1 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_1 = None
        x_10 = x_9 + input_8
        x_9 = input_8 = None
        layer_norm_3 = torch.nn.functional.layer_norm(
            x_10,
            (128,),
            l_self_modules_features_modules_1_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_1_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_1_modules_1_modules_norm1_parameters_weight_ = (
            l_self_modules_features_modules_1_modules_1_modules_norm1_parameters_bias_
        ) = None
        relative_position_bias_3 = l_self_modules_features_modules_1_modules_1_modules_attn_parameters_relative_position_bias_table_[
            l_self_modules_features_modules_1_modules_1_modules_attn_buffers_relative_position_index_
        ]
        l_self_modules_features_modules_1_modules_1_modules_attn_parameters_relative_position_bias_table_ = l_self_modules_features_modules_1_modules_1_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_4 = relative_position_bias_3.view(49, 49, -1)
        relative_position_bias_3 = None
        permute_5 = relative_position_bias_4.permute(2, 0, 1)
        relative_position_bias_4 = None
        contiguous_2 = permute_5.contiguous()
        permute_5 = None
        relative_position_bias_5 = contiguous_2.unsqueeze(0)
        contiguous_2 = None
        x_11 = torch._C._nn.pad(layer_norm_3, (0, 0, 0, 0, 0, 0), "constant", None)
        layer_norm_3 = None
        x_12 = torch.roll(x_11, shifts=(-3, -3), dims=(1, 2))
        x_11 = None
        x_13 = x_12.view(1, 8, 7, 8, 7, 128)
        x_12 = None
        permute_6 = x_13.permute(0, 1, 3, 2, 4, 5)
        x_13 = None
        x_14 = permute_6.reshape(64, 49, 128)
        permute_6 = None
        qkv_2 = torch._C._nn.linear(
            x_14,
            l_self_modules_features_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_features_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_,
        )
        l_self_modules_features_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_features_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_5 = qkv_2.reshape(64, 49, 3, 4, 32)
        qkv_2 = None
        qkv_3 = reshape_5.permute(2, 0, 3, 1, 4)
        reshape_5 = None
        q_2 = qkv_3[0]
        k_1 = qkv_3[1]
        v_1 = qkv_3[2]
        qkv_3 = None
        q_3 = q_2 * 0.1767766952966369
        q_2 = None
        transpose_2 = k_1.transpose(-2, -1)
        k_1 = None
        attn_4 = q_3.matmul(transpose_2)
        q_3 = transpose_2 = None
        attn_5 = attn_4 + relative_position_bias_5
        attn_4 = relative_position_bias_5 = None
        attn_mask = x_14.new_zeros((56, 56))
        x_14 = None
        attn_mask[(slice(0, -7, None), slice(0, -7, None))] = 0
        setitem = attn_mask
        setitem = None
        attn_mask[(slice(0, -7, None), slice(-7, -3, None))] = 1
        setitem_1 = attn_mask
        setitem_1 = None
        attn_mask[(slice(0, -7, None), slice(-3, None, None))] = 2
        setitem_2 = attn_mask
        setitem_2 = None
        attn_mask[(slice(-7, -3, None), slice(0, -7, None))] = 3
        setitem_3 = attn_mask
        setitem_3 = None
        attn_mask[(slice(-7, -3, None), slice(-7, -3, None))] = 4
        setitem_4 = attn_mask
        setitem_4 = None
        attn_mask[(slice(-7, -3, None), slice(-3, None, None))] = 5
        setitem_5 = attn_mask
        setitem_5 = None
        attn_mask[(slice(-3, None, None), slice(0, -7, None))] = 6
        setitem_6 = attn_mask
        setitem_6 = None
        attn_mask[(slice(-3, None, None), slice(-7, -3, None))] = 7
        setitem_7 = attn_mask
        setitem_7 = None
        attn_mask[(slice(-3, None, None), slice(-3, None, None))] = 8
        setitem_8 = attn_mask
        setitem_8 = None
        attn_mask_1 = attn_mask.view(8, 7, 8, 7)
        attn_mask = None
        permute_8 = attn_mask_1.permute(0, 2, 1, 3)
        attn_mask_1 = None
        attn_mask_2 = permute_8.reshape(64, 49)
        permute_8 = None
        unsqueeze_2 = attn_mask_2.unsqueeze(1)
        unsqueeze_3 = attn_mask_2.unsqueeze(2)
        attn_mask_2 = None
        attn_mask_3 = unsqueeze_2 - unsqueeze_3
        unsqueeze_2 = unsqueeze_3 = None
        ne = attn_mask_3 != 0
        masked_fill = attn_mask_3.masked_fill(ne, -100.0)
        ne = None
        eq = attn_mask_3 == 0
        attn_mask_3 = None
        attn_mask_4 = masked_fill.masked_fill(eq, 0.0)
        masked_fill = eq = None
        attn_6 = attn_5.view(1, 64, 4, 49, 49)
        attn_5 = None
        unsqueeze_4 = attn_mask_4.unsqueeze(1)
        attn_mask_4 = None
        unsqueeze_5 = unsqueeze_4.unsqueeze(0)
        unsqueeze_4 = None
        attn_7 = attn_6 + unsqueeze_5
        attn_6 = unsqueeze_5 = None
        attn_8 = attn_7.view(-1, 4, 49, 49)
        attn_7 = None
        attn_9 = torch.nn.functional.softmax(attn_8, dim=-1)
        attn_8 = None
        attn_10 = torch.nn.functional.dropout(attn_9, p=0.0, training=False)
        attn_9 = None
        matmul_3 = attn_10.matmul(v_1)
        attn_10 = v_1 = None
        transpose_3 = matmul_3.transpose(1, 2)
        matmul_3 = None
        x_15 = transpose_3.reshape(64, 49, 128)
        transpose_3 = None
        x_16 = torch._C._nn.linear(
            x_15,
            l_self_modules_features_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_15 = l_self_modules_features_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_17 = torch.nn.functional.dropout(x_16, p=0.0, training=False)
        x_16 = None
        x_18 = x_17.view(1, 8, 8, 7, 7, 128)
        x_17 = None
        permute_9 = x_18.permute(0, 1, 3, 2, 4, 5)
        x_18 = None
        x_19 = permute_9.reshape(1, 56, 56, 128)
        permute_9 = None
        x_20 = torch.roll(x_19, shifts=(3, 3), dims=(1, 2))
        x_19 = None
        getitem_9 = x_20[
            (
                slice(None, None, None),
                slice(None, 56, None),
                slice(None, 56, None),
                slice(None, None, None),
            )
        ]
        x_20 = None
        x_21 = getitem_9.contiguous()
        getitem_9 = None
        _log_api_usage_once_2 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_2 = None
        x_22 = x_10 + x_21
        x_10 = x_21 = None
        layer_norm_4 = torch.nn.functional.layer_norm(
            x_22,
            (128,),
            l_self_modules_features_modules_1_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_1_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_1_modules_1_modules_norm2_parameters_weight_ = (
            l_self_modules_features_modules_1_modules_1_modules_norm2_parameters_bias_
        ) = None
        input_9 = torch._C._nn.linear(
            layer_norm_4,
            l_self_modules_features_modules_1_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_1_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        layer_norm_4 = l_self_modules_features_modules_1_modules_1_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_1_modules_1_modules_mlp_modules_0_parameters_bias_ = (None)
        input_10 = torch._C._nn.gelu(input_9, approximate="none")
        input_9 = None
        input_11 = torch.nn.functional.dropout(input_10, 0.0, False, False)
        input_10 = None
        input_12 = torch._C._nn.linear(
            input_11,
            l_self_modules_features_modules_1_modules_1_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_1_modules_1_modules_mlp_modules_3_parameters_bias_,
        )
        input_11 = l_self_modules_features_modules_1_modules_1_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_1_modules_1_modules_mlp_modules_3_parameters_bias_ = (None)
        input_13 = torch.nn.functional.dropout(input_12, 0.0, False, False)
        input_12 = None
        _log_api_usage_once_3 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_3 = None
        x_23 = x_22 + input_13
        x_22 = input_13 = None
        x_24 = torch._C._nn.pad(x_23, (0, 0, 0, 0, 0, 0), "constant", None)
        x_23 = None
        x0 = x_24[
            (Ellipsis, slice(0, None, 2), slice(0, None, 2), slice(None, None, None))
        ]
        x1 = x_24[
            (Ellipsis, slice(1, None, 2), slice(0, None, 2), slice(None, None, None))
        ]
        x2 = x_24[
            (Ellipsis, slice(0, None, 2), slice(1, None, 2), slice(None, None, None))
        ]
        x3 = x_24[
            (Ellipsis, slice(1, None, 2), slice(1, None, 2), slice(None, None, None))
        ]
        x_24 = None
        x_25 = torch.cat([x0, x1, x2, x3], -1)
        x0 = x1 = x2 = x3 = None
        x_26 = torch.nn.functional.layer_norm(
            x_25,
            (512,),
            l_self_modules_features_modules_2_modules_norm_parameters_weight_,
            l_self_modules_features_modules_2_modules_norm_parameters_bias_,
            1e-05,
        )
        x_25 = (
            l_self_modules_features_modules_2_modules_norm_parameters_weight_
        ) = l_self_modules_features_modules_2_modules_norm_parameters_bias_ = None
        x_27 = torch._C._nn.linear(
            x_26,
            l_self_modules_features_modules_2_modules_reduction_parameters_weight_,
            None,
        )
        x_26 = (
            l_self_modules_features_modules_2_modules_reduction_parameters_weight_
        ) = None
        layer_norm_6 = torch.nn.functional.layer_norm(
            x_27,
            (256,),
            l_self_modules_features_modules_3_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_3_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_3_modules_0_modules_norm1_parameters_weight_ = (
            l_self_modules_features_modules_3_modules_0_modules_norm1_parameters_bias_
        ) = None
        relative_position_bias_6 = l_self_modules_features_modules_3_modules_0_modules_attn_parameters_relative_position_bias_table_[
            l_self_modules_features_modules_3_modules_0_modules_attn_buffers_relative_position_index_
        ]
        l_self_modules_features_modules_3_modules_0_modules_attn_parameters_relative_position_bias_table_ = l_self_modules_features_modules_3_modules_0_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_7 = relative_position_bias_6.view(49, 49, -1)
        relative_position_bias_6 = None
        permute_10 = relative_position_bias_7.permute(2, 0, 1)
        relative_position_bias_7 = None
        contiguous_4 = permute_10.contiguous()
        permute_10 = None
        relative_position_bias_8 = contiguous_4.unsqueeze(0)
        contiguous_4 = None
        x_28 = torch._C._nn.pad(layer_norm_6, (0, 0, 0, 0, 0, 0), "constant", None)
        layer_norm_6 = None
        x_29 = x_28.view(1, 4, 7, 4, 7, 256)
        x_28 = None
        permute_11 = x_29.permute(0, 1, 3, 2, 4, 5)
        x_29 = None
        x_30 = permute_11.reshape(16, 49, 256)
        permute_11 = None
        qkv_4 = torch._C._nn.linear(
            x_30,
            l_self_modules_features_modules_3_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_features_modules_3_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        x_30 = l_self_modules_features_modules_3_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_features_modules_3_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_10 = qkv_4.reshape(16, 49, 3, 8, 32)
        qkv_4 = None
        qkv_5 = reshape_10.permute(2, 0, 3, 1, 4)
        reshape_10 = None
        q_4 = qkv_5[0]
        k_2 = qkv_5[1]
        v_2 = qkv_5[2]
        qkv_5 = None
        q_5 = q_4 * 0.1767766952966369
        q_4 = None
        transpose_4 = k_2.transpose(-2, -1)
        k_2 = None
        attn_11 = q_5.matmul(transpose_4)
        q_5 = transpose_4 = None
        attn_12 = attn_11 + relative_position_bias_8
        attn_11 = relative_position_bias_8 = None
        attn_13 = torch.nn.functional.softmax(attn_12, dim=-1)
        attn_12 = None
        attn_14 = torch.nn.functional.dropout(attn_13, p=0.0, training=False)
        attn_13 = None
        matmul_5 = attn_14.matmul(v_2)
        attn_14 = v_2 = None
        transpose_5 = matmul_5.transpose(1, 2)
        matmul_5 = None
        x_31 = transpose_5.reshape(16, 49, 256)
        transpose_5 = None
        x_32 = torch._C._nn.linear(
            x_31,
            l_self_modules_features_modules_3_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_3_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_31 = l_self_modules_features_modules_3_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_3_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_33 = torch.nn.functional.dropout(x_32, p=0.0, training=False)
        x_32 = None
        x_34 = x_33.view(1, 4, 4, 7, 7, 256)
        x_33 = None
        permute_13 = x_34.permute(0, 1, 3, 2, 4, 5)
        x_34 = None
        x_35 = permute_13.reshape(1, 28, 28, 256)
        permute_13 = None
        getitem_18 = x_35[
            (
                slice(None, None, None),
                slice(None, 28, None),
                slice(None, 28, None),
                slice(None, None, None),
            )
        ]
        x_35 = None
        x_36 = getitem_18.contiguous()
        getitem_18 = None
        _log_api_usage_once_4 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_4 = None
        x_37 = x_27 + x_36
        x_27 = x_36 = None
        layer_norm_7 = torch.nn.functional.layer_norm(
            x_37,
            (256,),
            l_self_modules_features_modules_3_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_3_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_3_modules_0_modules_norm2_parameters_weight_ = (
            l_self_modules_features_modules_3_modules_0_modules_norm2_parameters_bias_
        ) = None
        input_14 = torch._C._nn.linear(
            layer_norm_7,
            l_self_modules_features_modules_3_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_3_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        layer_norm_7 = l_self_modules_features_modules_3_modules_0_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_3_modules_0_modules_mlp_modules_0_parameters_bias_ = (None)
        input_15 = torch._C._nn.gelu(input_14, approximate="none")
        input_14 = None
        input_16 = torch.nn.functional.dropout(input_15, 0.0, False, False)
        input_15 = None
        input_17 = torch._C._nn.linear(
            input_16,
            l_self_modules_features_modules_3_modules_0_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_3_modules_0_modules_mlp_modules_3_parameters_bias_,
        )
        input_16 = l_self_modules_features_modules_3_modules_0_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_3_modules_0_modules_mlp_modules_3_parameters_bias_ = (None)
        input_18 = torch.nn.functional.dropout(input_17, 0.0, False, False)
        input_17 = None
        _log_api_usage_once_5 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_5 = None
        x_38 = x_37 + input_18
        x_37 = input_18 = None
        layer_norm_8 = torch.nn.functional.layer_norm(
            x_38,
            (256,),
            l_self_modules_features_modules_3_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_3_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_3_modules_1_modules_norm1_parameters_weight_ = (
            l_self_modules_features_modules_3_modules_1_modules_norm1_parameters_bias_
        ) = None
        relative_position_bias_9 = l_self_modules_features_modules_3_modules_1_modules_attn_parameters_relative_position_bias_table_[
            l_self_modules_features_modules_3_modules_1_modules_attn_buffers_relative_position_index_
        ]
        l_self_modules_features_modules_3_modules_1_modules_attn_parameters_relative_position_bias_table_ = l_self_modules_features_modules_3_modules_1_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_10 = relative_position_bias_9.view(49, 49, -1)
        relative_position_bias_9 = None
        permute_14 = relative_position_bias_10.permute(2, 0, 1)
        relative_position_bias_10 = None
        contiguous_6 = permute_14.contiguous()
        permute_14 = None
        relative_position_bias_11 = contiguous_6.unsqueeze(0)
        contiguous_6 = None
        x_39 = torch._C._nn.pad(layer_norm_8, (0, 0, 0, 0, 0, 0), "constant", None)
        layer_norm_8 = None
        x_40 = torch.roll(x_39, shifts=(-3, -3), dims=(1, 2))
        x_39 = None
        x_41 = x_40.view(1, 4, 7, 4, 7, 256)
        x_40 = None
        permute_15 = x_41.permute(0, 1, 3, 2, 4, 5)
        x_41 = None
        x_42 = permute_15.reshape(16, 49, 256)
        permute_15 = None
        qkv_6 = torch._C._nn.linear(
            x_42,
            l_self_modules_features_modules_3_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_features_modules_3_modules_1_modules_attn_modules_qkv_parameters_bias_,
        )
        l_self_modules_features_modules_3_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_features_modules_3_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_14 = qkv_6.reshape(16, 49, 3, 8, 32)
        qkv_6 = None
        qkv_7 = reshape_14.permute(2, 0, 3, 1, 4)
        reshape_14 = None
        q_6 = qkv_7[0]
        k_3 = qkv_7[1]
        v_3 = qkv_7[2]
        qkv_7 = None
        q_7 = q_6 * 0.1767766952966369
        q_6 = None
        transpose_6 = k_3.transpose(-2, -1)
        k_3 = None
        attn_15 = q_7.matmul(transpose_6)
        q_7 = transpose_6 = None
        attn_16 = attn_15 + relative_position_bias_11
        attn_15 = relative_position_bias_11 = None
        attn_mask_5 = x_42.new_zeros((28, 28))
        x_42 = None
        attn_mask_5[(slice(0, -7, None), slice(0, -7, None))] = 0
        setitem_9 = attn_mask_5
        setitem_9 = None
        attn_mask_5[(slice(0, -7, None), slice(-7, -3, None))] = 1
        setitem_10 = attn_mask_5
        setitem_10 = None
        attn_mask_5[(slice(0, -7, None), slice(-3, None, None))] = 2
        setitem_11 = attn_mask_5
        setitem_11 = None
        attn_mask_5[(slice(-7, -3, None), slice(0, -7, None))] = 3
        setitem_12 = attn_mask_5
        setitem_12 = None
        attn_mask_5[(slice(-7, -3, None), slice(-7, -3, None))] = 4
        setitem_13 = attn_mask_5
        setitem_13 = None
        attn_mask_5[(slice(-7, -3, None), slice(-3, None, None))] = 5
        setitem_14 = attn_mask_5
        setitem_14 = None
        attn_mask_5[(slice(-3, None, None), slice(0, -7, None))] = 6
        setitem_15 = attn_mask_5
        setitem_15 = None
        attn_mask_5[(slice(-3, None, None), slice(-7, -3, None))] = 7
        setitem_16 = attn_mask_5
        setitem_16 = None
        attn_mask_5[(slice(-3, None, None), slice(-3, None, None))] = 8
        setitem_17 = attn_mask_5
        setitem_17 = None
        attn_mask_6 = attn_mask_5.view(4, 7, 4, 7)
        attn_mask_5 = None
        permute_17 = attn_mask_6.permute(0, 2, 1, 3)
        attn_mask_6 = None
        attn_mask_7 = permute_17.reshape(16, 49)
        permute_17 = None
        unsqueeze_8 = attn_mask_7.unsqueeze(1)
        unsqueeze_9 = attn_mask_7.unsqueeze(2)
        attn_mask_7 = None
        attn_mask_8 = unsqueeze_8 - unsqueeze_9
        unsqueeze_8 = unsqueeze_9 = None
        ne_1 = attn_mask_8 != 0
        masked_fill_2 = attn_mask_8.masked_fill(ne_1, -100.0)
        ne_1 = None
        eq_1 = attn_mask_8 == 0
        attn_mask_8 = None
        attn_mask_9 = masked_fill_2.masked_fill(eq_1, 0.0)
        masked_fill_2 = eq_1 = None
        attn_17 = attn_16.view(1, 16, 8, 49, 49)
        attn_16 = None
        unsqueeze_10 = attn_mask_9.unsqueeze(1)
        attn_mask_9 = None
        unsqueeze_11 = unsqueeze_10.unsqueeze(0)
        unsqueeze_10 = None
        attn_18 = attn_17 + unsqueeze_11
        attn_17 = unsqueeze_11 = None
        attn_19 = attn_18.view(-1, 8, 49, 49)
        attn_18 = None
        attn_20 = torch.nn.functional.softmax(attn_19, dim=-1)
        attn_19 = None
        attn_21 = torch.nn.functional.dropout(attn_20, p=0.0, training=False)
        attn_20 = None
        matmul_7 = attn_21.matmul(v_3)
        attn_21 = v_3 = None
        transpose_7 = matmul_7.transpose(1, 2)
        matmul_7 = None
        x_43 = transpose_7.reshape(16, 49, 256)
        transpose_7 = None
        x_44 = torch._C._nn.linear(
            x_43,
            l_self_modules_features_modules_3_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_3_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_43 = l_self_modules_features_modules_3_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_3_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_45 = torch.nn.functional.dropout(x_44, p=0.0, training=False)
        x_44 = None
        x_46 = x_45.view(1, 4, 4, 7, 7, 256)
        x_45 = None
        permute_18 = x_46.permute(0, 1, 3, 2, 4, 5)
        x_46 = None
        x_47 = permute_18.reshape(1, 28, 28, 256)
        permute_18 = None
        x_48 = torch.roll(x_47, shifts=(3, 3), dims=(1, 2))
        x_47 = None
        getitem_23 = x_48[
            (
                slice(None, None, None),
                slice(None, 28, None),
                slice(None, 28, None),
                slice(None, None, None),
            )
        ]
        x_48 = None
        x_49 = getitem_23.contiguous()
        getitem_23 = None
        _log_api_usage_once_6 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_6 = None
        x_50 = x_38 + x_49
        x_38 = x_49 = None
        layer_norm_9 = torch.nn.functional.layer_norm(
            x_50,
            (256,),
            l_self_modules_features_modules_3_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_3_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_3_modules_1_modules_norm2_parameters_weight_ = (
            l_self_modules_features_modules_3_modules_1_modules_norm2_parameters_bias_
        ) = None
        input_19 = torch._C._nn.linear(
            layer_norm_9,
            l_self_modules_features_modules_3_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_3_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        layer_norm_9 = l_self_modules_features_modules_3_modules_1_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_3_modules_1_modules_mlp_modules_0_parameters_bias_ = (None)
        input_20 = torch._C._nn.gelu(input_19, approximate="none")
        input_19 = None
        input_21 = torch.nn.functional.dropout(input_20, 0.0, False, False)
        input_20 = None
        input_22 = torch._C._nn.linear(
            input_21,
            l_self_modules_features_modules_3_modules_1_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_3_modules_1_modules_mlp_modules_3_parameters_bias_,
        )
        input_21 = l_self_modules_features_modules_3_modules_1_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_3_modules_1_modules_mlp_modules_3_parameters_bias_ = (None)
        input_23 = torch.nn.functional.dropout(input_22, 0.0, False, False)
        input_22 = None
        _log_api_usage_once_7 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_7 = None
        x_51 = x_50 + input_23
        x_50 = input_23 = None
        x_52 = torch._C._nn.pad(x_51, (0, 0, 0, 0, 0, 0), "constant", None)
        x_51 = None
        x0_1 = x_52[
            (Ellipsis, slice(0, None, 2), slice(0, None, 2), slice(None, None, None))
        ]
        x1_1 = x_52[
            (Ellipsis, slice(1, None, 2), slice(0, None, 2), slice(None, None, None))
        ]
        x2_1 = x_52[
            (Ellipsis, slice(0, None, 2), slice(1, None, 2), slice(None, None, None))
        ]
        x3_1 = x_52[
            (Ellipsis, slice(1, None, 2), slice(1, None, 2), slice(None, None, None))
        ]
        x_52 = None
        x_53 = torch.cat([x0_1, x1_1, x2_1, x3_1], -1)
        x0_1 = x1_1 = x2_1 = x3_1 = None
        x_54 = torch.nn.functional.layer_norm(
            x_53,
            (1024,),
            l_self_modules_features_modules_4_modules_norm_parameters_weight_,
            l_self_modules_features_modules_4_modules_norm_parameters_bias_,
            1e-05,
        )
        x_53 = (
            l_self_modules_features_modules_4_modules_norm_parameters_weight_
        ) = l_self_modules_features_modules_4_modules_norm_parameters_bias_ = None
        x_55 = torch._C._nn.linear(
            x_54,
            l_self_modules_features_modules_4_modules_reduction_parameters_weight_,
            None,
        )
        x_54 = (
            l_self_modules_features_modules_4_modules_reduction_parameters_weight_
        ) = None
        layer_norm_11 = torch.nn.functional.layer_norm(
            x_55,
            (512,),
            l_self_modules_features_modules_5_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_5_modules_0_modules_norm1_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_0_modules_norm1_parameters_bias_
        ) = None
        relative_position_bias_12 = l_self_modules_features_modules_5_modules_0_modules_attn_parameters_relative_position_bias_table_[
            l_self_modules_features_modules_5_modules_0_modules_attn_buffers_relative_position_index_
        ]
        l_self_modules_features_modules_5_modules_0_modules_attn_parameters_relative_position_bias_table_ = l_self_modules_features_modules_5_modules_0_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_13 = relative_position_bias_12.view(49, 49, -1)
        relative_position_bias_12 = None
        permute_19 = relative_position_bias_13.permute(2, 0, 1)
        relative_position_bias_13 = None
        contiguous_8 = permute_19.contiguous()
        permute_19 = None
        relative_position_bias_14 = contiguous_8.unsqueeze(0)
        contiguous_8 = None
        x_56 = torch._C._nn.pad(layer_norm_11, (0, 0, 0, 0, 0, 0), "constant", None)
        layer_norm_11 = None
        x_57 = x_56.view(1, 2, 7, 2, 7, 512)
        x_56 = None
        permute_20 = x_57.permute(0, 1, 3, 2, 4, 5)
        x_57 = None
        x_58 = permute_20.reshape(4, 49, 512)
        permute_20 = None
        qkv_8 = torch._C._nn.linear(
            x_58,
            l_self_modules_features_modules_5_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        x_58 = l_self_modules_features_modules_5_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_19 = qkv_8.reshape(4, 49, 3, 16, 32)
        qkv_8 = None
        qkv_9 = reshape_19.permute(2, 0, 3, 1, 4)
        reshape_19 = None
        q_8 = qkv_9[0]
        k_4 = qkv_9[1]
        v_4 = qkv_9[2]
        qkv_9 = None
        q_9 = q_8 * 0.1767766952966369
        q_8 = None
        transpose_8 = k_4.transpose(-2, -1)
        k_4 = None
        attn_22 = q_9.matmul(transpose_8)
        q_9 = transpose_8 = None
        attn_23 = attn_22 + relative_position_bias_14
        attn_22 = relative_position_bias_14 = None
        attn_24 = torch.nn.functional.softmax(attn_23, dim=-1)
        attn_23 = None
        attn_25 = torch.nn.functional.dropout(attn_24, p=0.0, training=False)
        attn_24 = None
        matmul_9 = attn_25.matmul(v_4)
        attn_25 = v_4 = None
        transpose_9 = matmul_9.transpose(1, 2)
        matmul_9 = None
        x_59 = transpose_9.reshape(4, 49, 512)
        transpose_9 = None
        x_60 = torch._C._nn.linear(
            x_59,
            l_self_modules_features_modules_5_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_59 = l_self_modules_features_modules_5_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_61 = torch.nn.functional.dropout(x_60, p=0.0, training=False)
        x_60 = None
        x_62 = x_61.view(1, 2, 2, 7, 7, 512)
        x_61 = None
        permute_22 = x_62.permute(0, 1, 3, 2, 4, 5)
        x_62 = None
        x_63 = permute_22.reshape(1, 14, 14, 512)
        permute_22 = None
        getitem_32 = x_63[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_63 = None
        x_64 = getitem_32.contiguous()
        getitem_32 = None
        _log_api_usage_once_8 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_8 = None
        x_65 = x_55 + x_64
        x_55 = x_64 = None
        layer_norm_12 = torch.nn.functional.layer_norm(
            x_65,
            (512,),
            l_self_modules_features_modules_5_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_5_modules_0_modules_norm2_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_0_modules_norm2_parameters_bias_
        ) = None
        input_24 = torch._C._nn.linear(
            layer_norm_12,
            l_self_modules_features_modules_5_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        layer_norm_12 = l_self_modules_features_modules_5_modules_0_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_mlp_modules_0_parameters_bias_ = (None)
        input_25 = torch._C._nn.gelu(input_24, approximate="none")
        input_24 = None
        input_26 = torch.nn.functional.dropout(input_25, 0.0, False, False)
        input_25 = None
        input_27 = torch._C._nn.linear(
            input_26,
            l_self_modules_features_modules_5_modules_0_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_mlp_modules_3_parameters_bias_,
        )
        input_26 = l_self_modules_features_modules_5_modules_0_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_mlp_modules_3_parameters_bias_ = (None)
        input_28 = torch.nn.functional.dropout(input_27, 0.0, False, False)
        input_27 = None
        _log_api_usage_once_9 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_9 = None
        x_66 = x_65 + input_28
        x_65 = input_28 = None
        layer_norm_13 = torch.nn.functional.layer_norm(
            x_66,
            (512,),
            l_self_modules_features_modules_5_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_5_modules_1_modules_norm1_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_1_modules_norm1_parameters_bias_
        ) = None
        relative_position_bias_15 = l_self_modules_features_modules_5_modules_1_modules_attn_parameters_relative_position_bias_table_[
            l_self_modules_features_modules_5_modules_1_modules_attn_buffers_relative_position_index_
        ]
        l_self_modules_features_modules_5_modules_1_modules_attn_parameters_relative_position_bias_table_ = l_self_modules_features_modules_5_modules_1_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_16 = relative_position_bias_15.view(49, 49, -1)
        relative_position_bias_15 = None
        permute_23 = relative_position_bias_16.permute(2, 0, 1)
        relative_position_bias_16 = None
        contiguous_10 = permute_23.contiguous()
        permute_23 = None
        relative_position_bias_17 = contiguous_10.unsqueeze(0)
        contiguous_10 = None
        x_67 = torch._C._nn.pad(layer_norm_13, (0, 0, 0, 0, 0, 0), "constant", None)
        layer_norm_13 = None
        x_68 = torch.roll(x_67, shifts=(-3, -3), dims=(1, 2))
        x_67 = None
        x_69 = x_68.view(1, 2, 7, 2, 7, 512)
        x_68 = None
        permute_24 = x_69.permute(0, 1, 3, 2, 4, 5)
        x_69 = None
        x_70 = permute_24.reshape(4, 49, 512)
        permute_24 = None
        qkv_10 = torch._C._nn.linear(
            x_70,
            l_self_modules_features_modules_5_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_attn_modules_qkv_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_23 = qkv_10.reshape(4, 49, 3, 16, 32)
        qkv_10 = None
        qkv_11 = reshape_23.permute(2, 0, 3, 1, 4)
        reshape_23 = None
        q_10 = qkv_11[0]
        k_5 = qkv_11[1]
        v_5 = qkv_11[2]
        qkv_11 = None
        q_11 = q_10 * 0.1767766952966369
        q_10 = None
        transpose_10 = k_5.transpose(-2, -1)
        k_5 = None
        attn_26 = q_11.matmul(transpose_10)
        q_11 = transpose_10 = None
        attn_27 = attn_26 + relative_position_bias_17
        attn_26 = relative_position_bias_17 = None
        attn_mask_10 = x_70.new_zeros((14, 14))
        x_70 = None
        attn_mask_10[(slice(0, -7, None), slice(0, -7, None))] = 0
        setitem_18 = attn_mask_10
        setitem_18 = None
        attn_mask_10[(slice(0, -7, None), slice(-7, -3, None))] = 1
        setitem_19 = attn_mask_10
        setitem_19 = None
        attn_mask_10[(slice(0, -7, None), slice(-3, None, None))] = 2
        setitem_20 = attn_mask_10
        setitem_20 = None
        attn_mask_10[(slice(-7, -3, None), slice(0, -7, None))] = 3
        setitem_21 = attn_mask_10
        setitem_21 = None
        attn_mask_10[(slice(-7, -3, None), slice(-7, -3, None))] = 4
        setitem_22 = attn_mask_10
        setitem_22 = None
        attn_mask_10[(slice(-7, -3, None), slice(-3, None, None))] = 5
        setitem_23 = attn_mask_10
        setitem_23 = None
        attn_mask_10[(slice(-3, None, None), slice(0, -7, None))] = 6
        setitem_24 = attn_mask_10
        setitem_24 = None
        attn_mask_10[(slice(-3, None, None), slice(-7, -3, None))] = 7
        setitem_25 = attn_mask_10
        setitem_25 = None
        attn_mask_10[(slice(-3, None, None), slice(-3, None, None))] = 8
        setitem_26 = attn_mask_10
        setitem_26 = None
        attn_mask_11 = attn_mask_10.view(2, 7, 2, 7)
        attn_mask_10 = None
        permute_26 = attn_mask_11.permute(0, 2, 1, 3)
        attn_mask_11 = None
        attn_mask_12 = permute_26.reshape(4, 49)
        permute_26 = None
        unsqueeze_14 = attn_mask_12.unsqueeze(1)
        unsqueeze_15 = attn_mask_12.unsqueeze(2)
        attn_mask_12 = None
        attn_mask_13 = unsqueeze_14 - unsqueeze_15
        unsqueeze_14 = unsqueeze_15 = None
        ne_2 = attn_mask_13 != 0
        masked_fill_4 = attn_mask_13.masked_fill(ne_2, -100.0)
        ne_2 = None
        eq_2 = attn_mask_13 == 0
        attn_mask_13 = None
        attn_mask_14 = masked_fill_4.masked_fill(eq_2, 0.0)
        masked_fill_4 = eq_2 = None
        attn_28 = attn_27.view(1, 4, 16, 49, 49)
        attn_27 = None
        unsqueeze_16 = attn_mask_14.unsqueeze(1)
        attn_mask_14 = None
        unsqueeze_17 = unsqueeze_16.unsqueeze(0)
        unsqueeze_16 = None
        attn_29 = attn_28 + unsqueeze_17
        attn_28 = unsqueeze_17 = None
        attn_30 = attn_29.view(-1, 16, 49, 49)
        attn_29 = None
        attn_31 = torch.nn.functional.softmax(attn_30, dim=-1)
        attn_30 = None
        attn_32 = torch.nn.functional.dropout(attn_31, p=0.0, training=False)
        attn_31 = None
        matmul_11 = attn_32.matmul(v_5)
        attn_32 = v_5 = None
        transpose_11 = matmul_11.transpose(1, 2)
        matmul_11 = None
        x_71 = transpose_11.reshape(4, 49, 512)
        transpose_11 = None
        x_72 = torch._C._nn.linear(
            x_71,
            l_self_modules_features_modules_5_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_71 = l_self_modules_features_modules_5_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_73 = torch.nn.functional.dropout(x_72, p=0.0, training=False)
        x_72 = None
        x_74 = x_73.view(1, 2, 2, 7, 7, 512)
        x_73 = None
        permute_27 = x_74.permute(0, 1, 3, 2, 4, 5)
        x_74 = None
        x_75 = permute_27.reshape(1, 14, 14, 512)
        permute_27 = None
        x_76 = torch.roll(x_75, shifts=(3, 3), dims=(1, 2))
        x_75 = None
        getitem_37 = x_76[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_76 = None
        x_77 = getitem_37.contiguous()
        getitem_37 = None
        _log_api_usage_once_10 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_10 = None
        x_78 = x_66 + x_77
        x_66 = x_77 = None
        layer_norm_14 = torch.nn.functional.layer_norm(
            x_78,
            (512,),
            l_self_modules_features_modules_5_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_5_modules_1_modules_norm2_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_1_modules_norm2_parameters_bias_
        ) = None
        input_29 = torch._C._nn.linear(
            layer_norm_14,
            l_self_modules_features_modules_5_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        layer_norm_14 = l_self_modules_features_modules_5_modules_1_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_mlp_modules_0_parameters_bias_ = (None)
        input_30 = torch._C._nn.gelu(input_29, approximate="none")
        input_29 = None
        input_31 = torch.nn.functional.dropout(input_30, 0.0, False, False)
        input_30 = None
        input_32 = torch._C._nn.linear(
            input_31,
            l_self_modules_features_modules_5_modules_1_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_mlp_modules_3_parameters_bias_,
        )
        input_31 = l_self_modules_features_modules_5_modules_1_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_mlp_modules_3_parameters_bias_ = (None)
        input_33 = torch.nn.functional.dropout(input_32, 0.0, False, False)
        input_32 = None
        _log_api_usage_once_11 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_11 = None
        x_79 = x_78 + input_33
        x_78 = input_33 = None
        layer_norm_15 = torch.nn.functional.layer_norm(
            x_79,
            (512,),
            l_self_modules_features_modules_5_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_5_modules_2_modules_norm1_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_2_modules_norm1_parameters_bias_
        ) = None
        relative_position_bias_18 = l_self_modules_features_modules_5_modules_2_modules_attn_parameters_relative_position_bias_table_[
            l_self_modules_features_modules_5_modules_2_modules_attn_buffers_relative_position_index_
        ]
        l_self_modules_features_modules_5_modules_2_modules_attn_parameters_relative_position_bias_table_ = l_self_modules_features_modules_5_modules_2_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_19 = relative_position_bias_18.view(49, 49, -1)
        relative_position_bias_18 = None
        permute_28 = relative_position_bias_19.permute(2, 0, 1)
        relative_position_bias_19 = None
        contiguous_12 = permute_28.contiguous()
        permute_28 = None
        relative_position_bias_20 = contiguous_12.unsqueeze(0)
        contiguous_12 = None
        x_80 = torch._C._nn.pad(layer_norm_15, (0, 0, 0, 0, 0, 0), "constant", None)
        layer_norm_15 = None
        x_81 = x_80.view(1, 2, 7, 2, 7, 512)
        x_80 = None
        permute_29 = x_81.permute(0, 1, 3, 2, 4, 5)
        x_81 = None
        x_82 = permute_29.reshape(4, 49, 512)
        permute_29 = None
        qkv_12 = torch._C._nn.linear(
            x_82,
            l_self_modules_features_modules_5_modules_2_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_attn_modules_qkv_parameters_bias_,
        )
        x_82 = l_self_modules_features_modules_5_modules_2_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_28 = qkv_12.reshape(4, 49, 3, 16, 32)
        qkv_12 = None
        qkv_13 = reshape_28.permute(2, 0, 3, 1, 4)
        reshape_28 = None
        q_12 = qkv_13[0]
        k_6 = qkv_13[1]
        v_6 = qkv_13[2]
        qkv_13 = None
        q_13 = q_12 * 0.1767766952966369
        q_12 = None
        transpose_12 = k_6.transpose(-2, -1)
        k_6 = None
        attn_33 = q_13.matmul(transpose_12)
        q_13 = transpose_12 = None
        attn_34 = attn_33 + relative_position_bias_20
        attn_33 = relative_position_bias_20 = None
        attn_35 = torch.nn.functional.softmax(attn_34, dim=-1)
        attn_34 = None
        attn_36 = torch.nn.functional.dropout(attn_35, p=0.0, training=False)
        attn_35 = None
        matmul_13 = attn_36.matmul(v_6)
        attn_36 = v_6 = None
        transpose_13 = matmul_13.transpose(1, 2)
        matmul_13 = None
        x_83 = transpose_13.reshape(4, 49, 512)
        transpose_13 = None
        x_84 = torch._C._nn.linear(
            x_83,
            l_self_modules_features_modules_5_modules_2_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_attn_modules_proj_parameters_bias_,
        )
        x_83 = l_self_modules_features_modules_5_modules_2_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_attn_modules_proj_parameters_bias_ = (None)
        x_85 = torch.nn.functional.dropout(x_84, p=0.0, training=False)
        x_84 = None
        x_86 = x_85.view(1, 2, 2, 7, 7, 512)
        x_85 = None
        permute_31 = x_86.permute(0, 1, 3, 2, 4, 5)
        x_86 = None
        x_87 = permute_31.reshape(1, 14, 14, 512)
        permute_31 = None
        getitem_42 = x_87[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_87 = None
        x_88 = getitem_42.contiguous()
        getitem_42 = None
        _log_api_usage_once_12 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_12 = None
        x_89 = x_79 + x_88
        x_79 = x_88 = None
        layer_norm_16 = torch.nn.functional.layer_norm(
            x_89,
            (512,),
            l_self_modules_features_modules_5_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_5_modules_2_modules_norm2_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_2_modules_norm2_parameters_bias_
        ) = None
        input_34 = torch._C._nn.linear(
            layer_norm_16,
            l_self_modules_features_modules_5_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        layer_norm_16 = l_self_modules_features_modules_5_modules_2_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_mlp_modules_0_parameters_bias_ = (None)
        input_35 = torch._C._nn.gelu(input_34, approximate="none")
        input_34 = None
        input_36 = torch.nn.functional.dropout(input_35, 0.0, False, False)
        input_35 = None
        input_37 = torch._C._nn.linear(
            input_36,
            l_self_modules_features_modules_5_modules_2_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_mlp_modules_3_parameters_bias_,
        )
        input_36 = l_self_modules_features_modules_5_modules_2_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_mlp_modules_3_parameters_bias_ = (None)
        input_38 = torch.nn.functional.dropout(input_37, 0.0, False, False)
        input_37 = None
        _log_api_usage_once_13 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_13 = None
        x_90 = x_89 + input_38
        x_89 = input_38 = None
        layer_norm_17 = torch.nn.functional.layer_norm(
            x_90,
            (512,),
            l_self_modules_features_modules_5_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_5_modules_3_modules_norm1_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_3_modules_norm1_parameters_bias_
        ) = None
        relative_position_bias_21 = l_self_modules_features_modules_5_modules_3_modules_attn_parameters_relative_position_bias_table_[
            l_self_modules_features_modules_5_modules_3_modules_attn_buffers_relative_position_index_
        ]
        l_self_modules_features_modules_5_modules_3_modules_attn_parameters_relative_position_bias_table_ = l_self_modules_features_modules_5_modules_3_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_22 = relative_position_bias_21.view(49, 49, -1)
        relative_position_bias_21 = None
        permute_32 = relative_position_bias_22.permute(2, 0, 1)
        relative_position_bias_22 = None
        contiguous_14 = permute_32.contiguous()
        permute_32 = None
        relative_position_bias_23 = contiguous_14.unsqueeze(0)
        contiguous_14 = None
        x_91 = torch._C._nn.pad(layer_norm_17, (0, 0, 0, 0, 0, 0), "constant", None)
        layer_norm_17 = None
        x_92 = torch.roll(x_91, shifts=(-3, -3), dims=(1, 2))
        x_91 = None
        x_93 = x_92.view(1, 2, 7, 2, 7, 512)
        x_92 = None
        permute_33 = x_93.permute(0, 1, 3, 2, 4, 5)
        x_93 = None
        x_94 = permute_33.reshape(4, 49, 512)
        permute_33 = None
        qkv_14 = torch._C._nn.linear(
            x_94,
            l_self_modules_features_modules_5_modules_3_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_attn_modules_qkv_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_3_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_32 = qkv_14.reshape(4, 49, 3, 16, 32)
        qkv_14 = None
        qkv_15 = reshape_32.permute(2, 0, 3, 1, 4)
        reshape_32 = None
        q_14 = qkv_15[0]
        k_7 = qkv_15[1]
        v_7 = qkv_15[2]
        qkv_15 = None
        q_15 = q_14 * 0.1767766952966369
        q_14 = None
        transpose_14 = k_7.transpose(-2, -1)
        k_7 = None
        attn_37 = q_15.matmul(transpose_14)
        q_15 = transpose_14 = None
        attn_38 = attn_37 + relative_position_bias_23
        attn_37 = relative_position_bias_23 = None
        attn_mask_15 = x_94.new_zeros((14, 14))
        x_94 = None
        attn_mask_15[(slice(0, -7, None), slice(0, -7, None))] = 0
        setitem_27 = attn_mask_15
        setitem_27 = None
        attn_mask_15[(slice(0, -7, None), slice(-7, -3, None))] = 1
        setitem_28 = attn_mask_15
        setitem_28 = None
        attn_mask_15[(slice(0, -7, None), slice(-3, None, None))] = 2
        setitem_29 = attn_mask_15
        setitem_29 = None
        attn_mask_15[(slice(-7, -3, None), slice(0, -7, None))] = 3
        setitem_30 = attn_mask_15
        setitem_30 = None
        attn_mask_15[(slice(-7, -3, None), slice(-7, -3, None))] = 4
        setitem_31 = attn_mask_15
        setitem_31 = None
        attn_mask_15[(slice(-7, -3, None), slice(-3, None, None))] = 5
        setitem_32 = attn_mask_15
        setitem_32 = None
        attn_mask_15[(slice(-3, None, None), slice(0, -7, None))] = 6
        setitem_33 = attn_mask_15
        setitem_33 = None
        attn_mask_15[(slice(-3, None, None), slice(-7, -3, None))] = 7
        setitem_34 = attn_mask_15
        setitem_34 = None
        attn_mask_15[(slice(-3, None, None), slice(-3, None, None))] = 8
        setitem_35 = attn_mask_15
        setitem_35 = None
        attn_mask_16 = attn_mask_15.view(2, 7, 2, 7)
        attn_mask_15 = None
        permute_35 = attn_mask_16.permute(0, 2, 1, 3)
        attn_mask_16 = None
        attn_mask_17 = permute_35.reshape(4, 49)
        permute_35 = None
        unsqueeze_20 = attn_mask_17.unsqueeze(1)
        unsqueeze_21 = attn_mask_17.unsqueeze(2)
        attn_mask_17 = None
        attn_mask_18 = unsqueeze_20 - unsqueeze_21
        unsqueeze_20 = unsqueeze_21 = None
        ne_3 = attn_mask_18 != 0
        masked_fill_6 = attn_mask_18.masked_fill(ne_3, -100.0)
        ne_3 = None
        eq_3 = attn_mask_18 == 0
        attn_mask_18 = None
        attn_mask_19 = masked_fill_6.masked_fill(eq_3, 0.0)
        masked_fill_6 = eq_3 = None
        attn_39 = attn_38.view(1, 4, 16, 49, 49)
        attn_38 = None
        unsqueeze_22 = attn_mask_19.unsqueeze(1)
        attn_mask_19 = None
        unsqueeze_23 = unsqueeze_22.unsqueeze(0)
        unsqueeze_22 = None
        attn_40 = attn_39 + unsqueeze_23
        attn_39 = unsqueeze_23 = None
        attn_41 = attn_40.view(-1, 16, 49, 49)
        attn_40 = None
        attn_42 = torch.nn.functional.softmax(attn_41, dim=-1)
        attn_41 = None
        attn_43 = torch.nn.functional.dropout(attn_42, p=0.0, training=False)
        attn_42 = None
        matmul_15 = attn_43.matmul(v_7)
        attn_43 = v_7 = None
        transpose_15 = matmul_15.transpose(1, 2)
        matmul_15 = None
        x_95 = transpose_15.reshape(4, 49, 512)
        transpose_15 = None
        x_96 = torch._C._nn.linear(
            x_95,
            l_self_modules_features_modules_5_modules_3_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_attn_modules_proj_parameters_bias_,
        )
        x_95 = l_self_modules_features_modules_5_modules_3_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_attn_modules_proj_parameters_bias_ = (None)
        x_97 = torch.nn.functional.dropout(x_96, p=0.0, training=False)
        x_96 = None
        x_98 = x_97.view(1, 2, 2, 7, 7, 512)
        x_97 = None
        permute_36 = x_98.permute(0, 1, 3, 2, 4, 5)
        x_98 = None
        x_99 = permute_36.reshape(1, 14, 14, 512)
        permute_36 = None
        x_100 = torch.roll(x_99, shifts=(3, 3), dims=(1, 2))
        x_99 = None
        getitem_47 = x_100[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_100 = None
        x_101 = getitem_47.contiguous()
        getitem_47 = None
        _log_api_usage_once_14 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_14 = None
        x_102 = x_90 + x_101
        x_90 = x_101 = None
        layer_norm_18 = torch.nn.functional.layer_norm(
            x_102,
            (512,),
            l_self_modules_features_modules_5_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_5_modules_3_modules_norm2_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_3_modules_norm2_parameters_bias_
        ) = None
        input_39 = torch._C._nn.linear(
            layer_norm_18,
            l_self_modules_features_modules_5_modules_3_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_mlp_modules_0_parameters_bias_,
        )
        layer_norm_18 = l_self_modules_features_modules_5_modules_3_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_mlp_modules_0_parameters_bias_ = (None)
        input_40 = torch._C._nn.gelu(input_39, approximate="none")
        input_39 = None
        input_41 = torch.nn.functional.dropout(input_40, 0.0, False, False)
        input_40 = None
        input_42 = torch._C._nn.linear(
            input_41,
            l_self_modules_features_modules_5_modules_3_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_mlp_modules_3_parameters_bias_,
        )
        input_41 = l_self_modules_features_modules_5_modules_3_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_mlp_modules_3_parameters_bias_ = (None)
        input_43 = torch.nn.functional.dropout(input_42, 0.0, False, False)
        input_42 = None
        _log_api_usage_once_15 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_15 = None
        x_103 = x_102 + input_43
        x_102 = input_43 = None
        layer_norm_19 = torch.nn.functional.layer_norm(
            x_103,
            (512,),
            l_self_modules_features_modules_5_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_5_modules_4_modules_norm1_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_4_modules_norm1_parameters_bias_
        ) = None
        relative_position_bias_24 = l_self_modules_features_modules_5_modules_4_modules_attn_parameters_relative_position_bias_table_[
            l_self_modules_features_modules_5_modules_4_modules_attn_buffers_relative_position_index_
        ]
        l_self_modules_features_modules_5_modules_4_modules_attn_parameters_relative_position_bias_table_ = l_self_modules_features_modules_5_modules_4_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_25 = relative_position_bias_24.view(49, 49, -1)
        relative_position_bias_24 = None
        permute_37 = relative_position_bias_25.permute(2, 0, 1)
        relative_position_bias_25 = None
        contiguous_16 = permute_37.contiguous()
        permute_37 = None
        relative_position_bias_26 = contiguous_16.unsqueeze(0)
        contiguous_16 = None
        x_104 = torch._C._nn.pad(layer_norm_19, (0, 0, 0, 0, 0, 0), "constant", None)
        layer_norm_19 = None
        x_105 = x_104.view(1, 2, 7, 2, 7, 512)
        x_104 = None
        permute_38 = x_105.permute(0, 1, 3, 2, 4, 5)
        x_105 = None
        x_106 = permute_38.reshape(4, 49, 512)
        permute_38 = None
        qkv_16 = torch._C._nn.linear(
            x_106,
            l_self_modules_features_modules_5_modules_4_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_attn_modules_qkv_parameters_bias_,
        )
        x_106 = l_self_modules_features_modules_5_modules_4_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_37 = qkv_16.reshape(4, 49, 3, 16, 32)
        qkv_16 = None
        qkv_17 = reshape_37.permute(2, 0, 3, 1, 4)
        reshape_37 = None
        q_16 = qkv_17[0]
        k_8 = qkv_17[1]
        v_8 = qkv_17[2]
        qkv_17 = None
        q_17 = q_16 * 0.1767766952966369
        q_16 = None
        transpose_16 = k_8.transpose(-2, -1)
        k_8 = None
        attn_44 = q_17.matmul(transpose_16)
        q_17 = transpose_16 = None
        attn_45 = attn_44 + relative_position_bias_26
        attn_44 = relative_position_bias_26 = None
        attn_46 = torch.nn.functional.softmax(attn_45, dim=-1)
        attn_45 = None
        attn_47 = torch.nn.functional.dropout(attn_46, p=0.0, training=False)
        attn_46 = None
        matmul_17 = attn_47.matmul(v_8)
        attn_47 = v_8 = None
        transpose_17 = matmul_17.transpose(1, 2)
        matmul_17 = None
        x_107 = transpose_17.reshape(4, 49, 512)
        transpose_17 = None
        x_108 = torch._C._nn.linear(
            x_107,
            l_self_modules_features_modules_5_modules_4_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_attn_modules_proj_parameters_bias_,
        )
        x_107 = l_self_modules_features_modules_5_modules_4_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_attn_modules_proj_parameters_bias_ = (None)
        x_109 = torch.nn.functional.dropout(x_108, p=0.0, training=False)
        x_108 = None
        x_110 = x_109.view(1, 2, 2, 7, 7, 512)
        x_109 = None
        permute_40 = x_110.permute(0, 1, 3, 2, 4, 5)
        x_110 = None
        x_111 = permute_40.reshape(1, 14, 14, 512)
        permute_40 = None
        getitem_52 = x_111[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_111 = None
        x_112 = getitem_52.contiguous()
        getitem_52 = None
        _log_api_usage_once_16 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_16 = None
        x_113 = x_103 + x_112
        x_103 = x_112 = None
        layer_norm_20 = torch.nn.functional.layer_norm(
            x_113,
            (512,),
            l_self_modules_features_modules_5_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_5_modules_4_modules_norm2_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_4_modules_norm2_parameters_bias_
        ) = None
        input_44 = torch._C._nn.linear(
            layer_norm_20,
            l_self_modules_features_modules_5_modules_4_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_mlp_modules_0_parameters_bias_,
        )
        layer_norm_20 = l_self_modules_features_modules_5_modules_4_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_mlp_modules_0_parameters_bias_ = (None)
        input_45 = torch._C._nn.gelu(input_44, approximate="none")
        input_44 = None
        input_46 = torch.nn.functional.dropout(input_45, 0.0, False, False)
        input_45 = None
        input_47 = torch._C._nn.linear(
            input_46,
            l_self_modules_features_modules_5_modules_4_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_mlp_modules_3_parameters_bias_,
        )
        input_46 = l_self_modules_features_modules_5_modules_4_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_mlp_modules_3_parameters_bias_ = (None)
        input_48 = torch.nn.functional.dropout(input_47, 0.0, False, False)
        input_47 = None
        _log_api_usage_once_17 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_17 = None
        x_114 = x_113 + input_48
        x_113 = input_48 = None
        layer_norm_21 = torch.nn.functional.layer_norm(
            x_114,
            (512,),
            l_self_modules_features_modules_5_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_5_modules_5_modules_norm1_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_5_modules_norm1_parameters_bias_
        ) = None
        relative_position_bias_27 = l_self_modules_features_modules_5_modules_5_modules_attn_parameters_relative_position_bias_table_[
            l_self_modules_features_modules_5_modules_5_modules_attn_buffers_relative_position_index_
        ]
        l_self_modules_features_modules_5_modules_5_modules_attn_parameters_relative_position_bias_table_ = l_self_modules_features_modules_5_modules_5_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_28 = relative_position_bias_27.view(49, 49, -1)
        relative_position_bias_27 = None
        permute_41 = relative_position_bias_28.permute(2, 0, 1)
        relative_position_bias_28 = None
        contiguous_18 = permute_41.contiguous()
        permute_41 = None
        relative_position_bias_29 = contiguous_18.unsqueeze(0)
        contiguous_18 = None
        x_115 = torch._C._nn.pad(layer_norm_21, (0, 0, 0, 0, 0, 0), "constant", None)
        layer_norm_21 = None
        x_116 = torch.roll(x_115, shifts=(-3, -3), dims=(1, 2))
        x_115 = None
        x_117 = x_116.view(1, 2, 7, 2, 7, 512)
        x_116 = None
        permute_42 = x_117.permute(0, 1, 3, 2, 4, 5)
        x_117 = None
        x_118 = permute_42.reshape(4, 49, 512)
        permute_42 = None
        qkv_18 = torch._C._nn.linear(
            x_118,
            l_self_modules_features_modules_5_modules_5_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_attn_modules_qkv_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_5_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_41 = qkv_18.reshape(4, 49, 3, 16, 32)
        qkv_18 = None
        qkv_19 = reshape_41.permute(2, 0, 3, 1, 4)
        reshape_41 = None
        q_18 = qkv_19[0]
        k_9 = qkv_19[1]
        v_9 = qkv_19[2]
        qkv_19 = None
        q_19 = q_18 * 0.1767766952966369
        q_18 = None
        transpose_18 = k_9.transpose(-2, -1)
        k_9 = None
        attn_48 = q_19.matmul(transpose_18)
        q_19 = transpose_18 = None
        attn_49 = attn_48 + relative_position_bias_29
        attn_48 = relative_position_bias_29 = None
        attn_mask_20 = x_118.new_zeros((14, 14))
        x_118 = None
        attn_mask_20[(slice(0, -7, None), slice(0, -7, None))] = 0
        setitem_36 = attn_mask_20
        setitem_36 = None
        attn_mask_20[(slice(0, -7, None), slice(-7, -3, None))] = 1
        setitem_37 = attn_mask_20
        setitem_37 = None
        attn_mask_20[(slice(0, -7, None), slice(-3, None, None))] = 2
        setitem_38 = attn_mask_20
        setitem_38 = None
        attn_mask_20[(slice(-7, -3, None), slice(0, -7, None))] = 3
        setitem_39 = attn_mask_20
        setitem_39 = None
        attn_mask_20[(slice(-7, -3, None), slice(-7, -3, None))] = 4
        setitem_40 = attn_mask_20
        setitem_40 = None
        attn_mask_20[(slice(-7, -3, None), slice(-3, None, None))] = 5
        setitem_41 = attn_mask_20
        setitem_41 = None
        attn_mask_20[(slice(-3, None, None), slice(0, -7, None))] = 6
        setitem_42 = attn_mask_20
        setitem_42 = None
        attn_mask_20[(slice(-3, None, None), slice(-7, -3, None))] = 7
        setitem_43 = attn_mask_20
        setitem_43 = None
        attn_mask_20[(slice(-3, None, None), slice(-3, None, None))] = 8
        setitem_44 = attn_mask_20
        setitem_44 = None
        attn_mask_21 = attn_mask_20.view(2, 7, 2, 7)
        attn_mask_20 = None
        permute_44 = attn_mask_21.permute(0, 2, 1, 3)
        attn_mask_21 = None
        attn_mask_22 = permute_44.reshape(4, 49)
        permute_44 = None
        unsqueeze_26 = attn_mask_22.unsqueeze(1)
        unsqueeze_27 = attn_mask_22.unsqueeze(2)
        attn_mask_22 = None
        attn_mask_23 = unsqueeze_26 - unsqueeze_27
        unsqueeze_26 = unsqueeze_27 = None
        ne_4 = attn_mask_23 != 0
        masked_fill_8 = attn_mask_23.masked_fill(ne_4, -100.0)
        ne_4 = None
        eq_4 = attn_mask_23 == 0
        attn_mask_23 = None
        attn_mask_24 = masked_fill_8.masked_fill(eq_4, 0.0)
        masked_fill_8 = eq_4 = None
        attn_50 = attn_49.view(1, 4, 16, 49, 49)
        attn_49 = None
        unsqueeze_28 = attn_mask_24.unsqueeze(1)
        attn_mask_24 = None
        unsqueeze_29 = unsqueeze_28.unsqueeze(0)
        unsqueeze_28 = None
        attn_51 = attn_50 + unsqueeze_29
        attn_50 = unsqueeze_29 = None
        attn_52 = attn_51.view(-1, 16, 49, 49)
        attn_51 = None
        attn_53 = torch.nn.functional.softmax(attn_52, dim=-1)
        attn_52 = None
        attn_54 = torch.nn.functional.dropout(attn_53, p=0.0, training=False)
        attn_53 = None
        matmul_19 = attn_54.matmul(v_9)
        attn_54 = v_9 = None
        transpose_19 = matmul_19.transpose(1, 2)
        matmul_19 = None
        x_119 = transpose_19.reshape(4, 49, 512)
        transpose_19 = None
        x_120 = torch._C._nn.linear(
            x_119,
            l_self_modules_features_modules_5_modules_5_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_attn_modules_proj_parameters_bias_,
        )
        x_119 = l_self_modules_features_modules_5_modules_5_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_attn_modules_proj_parameters_bias_ = (None)
        x_121 = torch.nn.functional.dropout(x_120, p=0.0, training=False)
        x_120 = None
        x_122 = x_121.view(1, 2, 2, 7, 7, 512)
        x_121 = None
        permute_45 = x_122.permute(0, 1, 3, 2, 4, 5)
        x_122 = None
        x_123 = permute_45.reshape(1, 14, 14, 512)
        permute_45 = None
        x_124 = torch.roll(x_123, shifts=(3, 3), dims=(1, 2))
        x_123 = None
        getitem_57 = x_124[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_124 = None
        x_125 = getitem_57.contiguous()
        getitem_57 = None
        _log_api_usage_once_18 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_18 = None
        x_126 = x_114 + x_125
        x_114 = x_125 = None
        layer_norm_22 = torch.nn.functional.layer_norm(
            x_126,
            (512,),
            l_self_modules_features_modules_5_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_5_modules_5_modules_norm2_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_5_modules_norm2_parameters_bias_
        ) = None
        input_49 = torch._C._nn.linear(
            layer_norm_22,
            l_self_modules_features_modules_5_modules_5_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_mlp_modules_0_parameters_bias_,
        )
        layer_norm_22 = l_self_modules_features_modules_5_modules_5_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_mlp_modules_0_parameters_bias_ = (None)
        input_50 = torch._C._nn.gelu(input_49, approximate="none")
        input_49 = None
        input_51 = torch.nn.functional.dropout(input_50, 0.0, False, False)
        input_50 = None
        input_52 = torch._C._nn.linear(
            input_51,
            l_self_modules_features_modules_5_modules_5_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_mlp_modules_3_parameters_bias_,
        )
        input_51 = l_self_modules_features_modules_5_modules_5_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_mlp_modules_3_parameters_bias_ = (None)
        input_53 = torch.nn.functional.dropout(input_52, 0.0, False, False)
        input_52 = None
        _log_api_usage_once_19 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_19 = None
        x_127 = x_126 + input_53
        x_126 = input_53 = None
        layer_norm_23 = torch.nn.functional.layer_norm(
            x_127,
            (512,),
            l_self_modules_features_modules_5_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_5_modules_6_modules_norm1_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_6_modules_norm1_parameters_bias_
        ) = None
        relative_position_bias_30 = l_self_modules_features_modules_5_modules_6_modules_attn_parameters_relative_position_bias_table_[
            l_self_modules_features_modules_5_modules_6_modules_attn_buffers_relative_position_index_
        ]
        l_self_modules_features_modules_5_modules_6_modules_attn_parameters_relative_position_bias_table_ = l_self_modules_features_modules_5_modules_6_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_31 = relative_position_bias_30.view(49, 49, -1)
        relative_position_bias_30 = None
        permute_46 = relative_position_bias_31.permute(2, 0, 1)
        relative_position_bias_31 = None
        contiguous_20 = permute_46.contiguous()
        permute_46 = None
        relative_position_bias_32 = contiguous_20.unsqueeze(0)
        contiguous_20 = None
        x_128 = torch._C._nn.pad(layer_norm_23, (0, 0, 0, 0, 0, 0), "constant", None)
        layer_norm_23 = None
        x_129 = x_128.view(1, 2, 7, 2, 7, 512)
        x_128 = None
        permute_47 = x_129.permute(0, 1, 3, 2, 4, 5)
        x_129 = None
        x_130 = permute_47.reshape(4, 49, 512)
        permute_47 = None
        qkv_20 = torch._C._nn.linear(
            x_130,
            l_self_modules_features_modules_5_modules_6_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_attn_modules_qkv_parameters_bias_,
        )
        x_130 = l_self_modules_features_modules_5_modules_6_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_46 = qkv_20.reshape(4, 49, 3, 16, 32)
        qkv_20 = None
        qkv_21 = reshape_46.permute(2, 0, 3, 1, 4)
        reshape_46 = None
        q_20 = qkv_21[0]
        k_10 = qkv_21[1]
        v_10 = qkv_21[2]
        qkv_21 = None
        q_21 = q_20 * 0.1767766952966369
        q_20 = None
        transpose_20 = k_10.transpose(-2, -1)
        k_10 = None
        attn_55 = q_21.matmul(transpose_20)
        q_21 = transpose_20 = None
        attn_56 = attn_55 + relative_position_bias_32
        attn_55 = relative_position_bias_32 = None
        attn_57 = torch.nn.functional.softmax(attn_56, dim=-1)
        attn_56 = None
        attn_58 = torch.nn.functional.dropout(attn_57, p=0.0, training=False)
        attn_57 = None
        matmul_21 = attn_58.matmul(v_10)
        attn_58 = v_10 = None
        transpose_21 = matmul_21.transpose(1, 2)
        matmul_21 = None
        x_131 = transpose_21.reshape(4, 49, 512)
        transpose_21 = None
        x_132 = torch._C._nn.linear(
            x_131,
            l_self_modules_features_modules_5_modules_6_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_attn_modules_proj_parameters_bias_,
        )
        x_131 = l_self_modules_features_modules_5_modules_6_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_attn_modules_proj_parameters_bias_ = (None)
        x_133 = torch.nn.functional.dropout(x_132, p=0.0, training=False)
        x_132 = None
        x_134 = x_133.view(1, 2, 2, 7, 7, 512)
        x_133 = None
        permute_49 = x_134.permute(0, 1, 3, 2, 4, 5)
        x_134 = None
        x_135 = permute_49.reshape(1, 14, 14, 512)
        permute_49 = None
        getitem_62 = x_135[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_135 = None
        x_136 = getitem_62.contiguous()
        getitem_62 = None
        _log_api_usage_once_20 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_20 = None
        x_137 = x_127 + x_136
        x_127 = x_136 = None
        layer_norm_24 = torch.nn.functional.layer_norm(
            x_137,
            (512,),
            l_self_modules_features_modules_5_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_5_modules_6_modules_norm2_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_6_modules_norm2_parameters_bias_
        ) = None
        input_54 = torch._C._nn.linear(
            layer_norm_24,
            l_self_modules_features_modules_5_modules_6_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_mlp_modules_0_parameters_bias_,
        )
        layer_norm_24 = l_self_modules_features_modules_5_modules_6_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_mlp_modules_0_parameters_bias_ = (None)
        input_55 = torch._C._nn.gelu(input_54, approximate="none")
        input_54 = None
        input_56 = torch.nn.functional.dropout(input_55, 0.0, False, False)
        input_55 = None
        input_57 = torch._C._nn.linear(
            input_56,
            l_self_modules_features_modules_5_modules_6_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_mlp_modules_3_parameters_bias_,
        )
        input_56 = l_self_modules_features_modules_5_modules_6_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_mlp_modules_3_parameters_bias_ = (None)
        input_58 = torch.nn.functional.dropout(input_57, 0.0, False, False)
        input_57 = None
        _log_api_usage_once_21 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_21 = None
        x_138 = x_137 + input_58
        x_137 = input_58 = None
        layer_norm_25 = torch.nn.functional.layer_norm(
            x_138,
            (512,),
            l_self_modules_features_modules_5_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_5_modules_7_modules_norm1_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_7_modules_norm1_parameters_bias_
        ) = None
        relative_position_bias_33 = l_self_modules_features_modules_5_modules_7_modules_attn_parameters_relative_position_bias_table_[
            l_self_modules_features_modules_5_modules_7_modules_attn_buffers_relative_position_index_
        ]
        l_self_modules_features_modules_5_modules_7_modules_attn_parameters_relative_position_bias_table_ = l_self_modules_features_modules_5_modules_7_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_34 = relative_position_bias_33.view(49, 49, -1)
        relative_position_bias_33 = None
        permute_50 = relative_position_bias_34.permute(2, 0, 1)
        relative_position_bias_34 = None
        contiguous_22 = permute_50.contiguous()
        permute_50 = None
        relative_position_bias_35 = contiguous_22.unsqueeze(0)
        contiguous_22 = None
        x_139 = torch._C._nn.pad(layer_norm_25, (0, 0, 0, 0, 0, 0), "constant", None)
        layer_norm_25 = None
        x_140 = torch.roll(x_139, shifts=(-3, -3), dims=(1, 2))
        x_139 = None
        x_141 = x_140.view(1, 2, 7, 2, 7, 512)
        x_140 = None
        permute_51 = x_141.permute(0, 1, 3, 2, 4, 5)
        x_141 = None
        x_142 = permute_51.reshape(4, 49, 512)
        permute_51 = None
        qkv_22 = torch._C._nn.linear(
            x_142,
            l_self_modules_features_modules_5_modules_7_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_attn_modules_qkv_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_7_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_50 = qkv_22.reshape(4, 49, 3, 16, 32)
        qkv_22 = None
        qkv_23 = reshape_50.permute(2, 0, 3, 1, 4)
        reshape_50 = None
        q_22 = qkv_23[0]
        k_11 = qkv_23[1]
        v_11 = qkv_23[2]
        qkv_23 = None
        q_23 = q_22 * 0.1767766952966369
        q_22 = None
        transpose_22 = k_11.transpose(-2, -1)
        k_11 = None
        attn_59 = q_23.matmul(transpose_22)
        q_23 = transpose_22 = None
        attn_60 = attn_59 + relative_position_bias_35
        attn_59 = relative_position_bias_35 = None
        attn_mask_25 = x_142.new_zeros((14, 14))
        x_142 = None
        attn_mask_25[(slice(0, -7, None), slice(0, -7, None))] = 0
        setitem_45 = attn_mask_25
        setitem_45 = None
        attn_mask_25[(slice(0, -7, None), slice(-7, -3, None))] = 1
        setitem_46 = attn_mask_25
        setitem_46 = None
        attn_mask_25[(slice(0, -7, None), slice(-3, None, None))] = 2
        setitem_47 = attn_mask_25
        setitem_47 = None
        attn_mask_25[(slice(-7, -3, None), slice(0, -7, None))] = 3
        setitem_48 = attn_mask_25
        setitem_48 = None
        attn_mask_25[(slice(-7, -3, None), slice(-7, -3, None))] = 4
        setitem_49 = attn_mask_25
        setitem_49 = None
        attn_mask_25[(slice(-7, -3, None), slice(-3, None, None))] = 5
        setitem_50 = attn_mask_25
        setitem_50 = None
        attn_mask_25[(slice(-3, None, None), slice(0, -7, None))] = 6
        setitem_51 = attn_mask_25
        setitem_51 = None
        attn_mask_25[(slice(-3, None, None), slice(-7, -3, None))] = 7
        setitem_52 = attn_mask_25
        setitem_52 = None
        attn_mask_25[(slice(-3, None, None), slice(-3, None, None))] = 8
        setitem_53 = attn_mask_25
        setitem_53 = None
        attn_mask_26 = attn_mask_25.view(2, 7, 2, 7)
        attn_mask_25 = None
        permute_53 = attn_mask_26.permute(0, 2, 1, 3)
        attn_mask_26 = None
        attn_mask_27 = permute_53.reshape(4, 49)
        permute_53 = None
        unsqueeze_32 = attn_mask_27.unsqueeze(1)
        unsqueeze_33 = attn_mask_27.unsqueeze(2)
        attn_mask_27 = None
        attn_mask_28 = unsqueeze_32 - unsqueeze_33
        unsqueeze_32 = unsqueeze_33 = None
        ne_5 = attn_mask_28 != 0
        masked_fill_10 = attn_mask_28.masked_fill(ne_5, -100.0)
        ne_5 = None
        eq_5 = attn_mask_28 == 0
        attn_mask_28 = None
        attn_mask_29 = masked_fill_10.masked_fill(eq_5, 0.0)
        masked_fill_10 = eq_5 = None
        attn_61 = attn_60.view(1, 4, 16, 49, 49)
        attn_60 = None
        unsqueeze_34 = attn_mask_29.unsqueeze(1)
        attn_mask_29 = None
        unsqueeze_35 = unsqueeze_34.unsqueeze(0)
        unsqueeze_34 = None
        attn_62 = attn_61 + unsqueeze_35
        attn_61 = unsqueeze_35 = None
        attn_63 = attn_62.view(-1, 16, 49, 49)
        attn_62 = None
        attn_64 = torch.nn.functional.softmax(attn_63, dim=-1)
        attn_63 = None
        attn_65 = torch.nn.functional.dropout(attn_64, p=0.0, training=False)
        attn_64 = None
        matmul_23 = attn_65.matmul(v_11)
        attn_65 = v_11 = None
        transpose_23 = matmul_23.transpose(1, 2)
        matmul_23 = None
        x_143 = transpose_23.reshape(4, 49, 512)
        transpose_23 = None
        x_144 = torch._C._nn.linear(
            x_143,
            l_self_modules_features_modules_5_modules_7_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_attn_modules_proj_parameters_bias_,
        )
        x_143 = l_self_modules_features_modules_5_modules_7_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_attn_modules_proj_parameters_bias_ = (None)
        x_145 = torch.nn.functional.dropout(x_144, p=0.0, training=False)
        x_144 = None
        x_146 = x_145.view(1, 2, 2, 7, 7, 512)
        x_145 = None
        permute_54 = x_146.permute(0, 1, 3, 2, 4, 5)
        x_146 = None
        x_147 = permute_54.reshape(1, 14, 14, 512)
        permute_54 = None
        x_148 = torch.roll(x_147, shifts=(3, 3), dims=(1, 2))
        x_147 = None
        getitem_67 = x_148[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_148 = None
        x_149 = getitem_67.contiguous()
        getitem_67 = None
        _log_api_usage_once_22 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_22 = None
        x_150 = x_138 + x_149
        x_138 = x_149 = None
        layer_norm_26 = torch.nn.functional.layer_norm(
            x_150,
            (512,),
            l_self_modules_features_modules_5_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_5_modules_7_modules_norm2_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_7_modules_norm2_parameters_bias_
        ) = None
        input_59 = torch._C._nn.linear(
            layer_norm_26,
            l_self_modules_features_modules_5_modules_7_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_mlp_modules_0_parameters_bias_,
        )
        layer_norm_26 = l_self_modules_features_modules_5_modules_7_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_mlp_modules_0_parameters_bias_ = (None)
        input_60 = torch._C._nn.gelu(input_59, approximate="none")
        input_59 = None
        input_61 = torch.nn.functional.dropout(input_60, 0.0, False, False)
        input_60 = None
        input_62 = torch._C._nn.linear(
            input_61,
            l_self_modules_features_modules_5_modules_7_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_mlp_modules_3_parameters_bias_,
        )
        input_61 = l_self_modules_features_modules_5_modules_7_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_mlp_modules_3_parameters_bias_ = (None)
        input_63 = torch.nn.functional.dropout(input_62, 0.0, False, False)
        input_62 = None
        _log_api_usage_once_23 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_23 = None
        x_151 = x_150 + input_63
        x_150 = input_63 = None
        layer_norm_27 = torch.nn.functional.layer_norm(
            x_151,
            (512,),
            l_self_modules_features_modules_5_modules_8_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_5_modules_8_modules_norm1_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_8_modules_norm1_parameters_bias_
        ) = None
        relative_position_bias_36 = l_self_modules_features_modules_5_modules_8_modules_attn_parameters_relative_position_bias_table_[
            l_self_modules_features_modules_5_modules_8_modules_attn_buffers_relative_position_index_
        ]
        l_self_modules_features_modules_5_modules_8_modules_attn_parameters_relative_position_bias_table_ = l_self_modules_features_modules_5_modules_8_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_37 = relative_position_bias_36.view(49, 49, -1)
        relative_position_bias_36 = None
        permute_55 = relative_position_bias_37.permute(2, 0, 1)
        relative_position_bias_37 = None
        contiguous_24 = permute_55.contiguous()
        permute_55 = None
        relative_position_bias_38 = contiguous_24.unsqueeze(0)
        contiguous_24 = None
        x_152 = torch._C._nn.pad(layer_norm_27, (0, 0, 0, 0, 0, 0), "constant", None)
        layer_norm_27 = None
        x_153 = x_152.view(1, 2, 7, 2, 7, 512)
        x_152 = None
        permute_56 = x_153.permute(0, 1, 3, 2, 4, 5)
        x_153 = None
        x_154 = permute_56.reshape(4, 49, 512)
        permute_56 = None
        qkv_24 = torch._C._nn.linear(
            x_154,
            l_self_modules_features_modules_5_modules_8_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_attn_modules_qkv_parameters_bias_,
        )
        x_154 = l_self_modules_features_modules_5_modules_8_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_features_modules_5_modules_8_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_55 = qkv_24.reshape(4, 49, 3, 16, 32)
        qkv_24 = None
        qkv_25 = reshape_55.permute(2, 0, 3, 1, 4)
        reshape_55 = None
        q_24 = qkv_25[0]
        k_12 = qkv_25[1]
        v_12 = qkv_25[2]
        qkv_25 = None
        q_25 = q_24 * 0.1767766952966369
        q_24 = None
        transpose_24 = k_12.transpose(-2, -1)
        k_12 = None
        attn_66 = q_25.matmul(transpose_24)
        q_25 = transpose_24 = None
        attn_67 = attn_66 + relative_position_bias_38
        attn_66 = relative_position_bias_38 = None
        attn_68 = torch.nn.functional.softmax(attn_67, dim=-1)
        attn_67 = None
        attn_69 = torch.nn.functional.dropout(attn_68, p=0.0, training=False)
        attn_68 = None
        matmul_25 = attn_69.matmul(v_12)
        attn_69 = v_12 = None
        transpose_25 = matmul_25.transpose(1, 2)
        matmul_25 = None
        x_155 = transpose_25.reshape(4, 49, 512)
        transpose_25 = None
        x_156 = torch._C._nn.linear(
            x_155,
            l_self_modules_features_modules_5_modules_8_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_attn_modules_proj_parameters_bias_,
        )
        x_155 = l_self_modules_features_modules_5_modules_8_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_5_modules_8_modules_attn_modules_proj_parameters_bias_ = (None)
        x_157 = torch.nn.functional.dropout(x_156, p=0.0, training=False)
        x_156 = None
        x_158 = x_157.view(1, 2, 2, 7, 7, 512)
        x_157 = None
        permute_58 = x_158.permute(0, 1, 3, 2, 4, 5)
        x_158 = None
        x_159 = permute_58.reshape(1, 14, 14, 512)
        permute_58 = None
        getitem_72 = x_159[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_159 = None
        x_160 = getitem_72.contiguous()
        getitem_72 = None
        _log_api_usage_once_24 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_24 = None
        x_161 = x_151 + x_160
        x_151 = x_160 = None
        layer_norm_28 = torch.nn.functional.layer_norm(
            x_161,
            (512,),
            l_self_modules_features_modules_5_modules_8_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_5_modules_8_modules_norm2_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_8_modules_norm2_parameters_bias_
        ) = None
        input_64 = torch._C._nn.linear(
            layer_norm_28,
            l_self_modules_features_modules_5_modules_8_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_mlp_modules_0_parameters_bias_,
        )
        layer_norm_28 = l_self_modules_features_modules_5_modules_8_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_8_modules_mlp_modules_0_parameters_bias_ = (None)
        input_65 = torch._C._nn.gelu(input_64, approximate="none")
        input_64 = None
        input_66 = torch.nn.functional.dropout(input_65, 0.0, False, False)
        input_65 = None
        input_67 = torch._C._nn.linear(
            input_66,
            l_self_modules_features_modules_5_modules_8_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_mlp_modules_3_parameters_bias_,
        )
        input_66 = l_self_modules_features_modules_5_modules_8_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_8_modules_mlp_modules_3_parameters_bias_ = (None)
        input_68 = torch.nn.functional.dropout(input_67, 0.0, False, False)
        input_67 = None
        _log_api_usage_once_25 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_25 = None
        x_162 = x_161 + input_68
        x_161 = input_68 = None
        layer_norm_29 = torch.nn.functional.layer_norm(
            x_162,
            (512,),
            l_self_modules_features_modules_5_modules_9_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_5_modules_9_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_5_modules_9_modules_norm1_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_9_modules_norm1_parameters_bias_
        ) = None
        relative_position_bias_39 = l_self_modules_features_modules_5_modules_9_modules_attn_parameters_relative_position_bias_table_[
            l_self_modules_features_modules_5_modules_9_modules_attn_buffers_relative_position_index_
        ]
        l_self_modules_features_modules_5_modules_9_modules_attn_parameters_relative_position_bias_table_ = l_self_modules_features_modules_5_modules_9_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_40 = relative_position_bias_39.view(49, 49, -1)
        relative_position_bias_39 = None
        permute_59 = relative_position_bias_40.permute(2, 0, 1)
        relative_position_bias_40 = None
        contiguous_26 = permute_59.contiguous()
        permute_59 = None
        relative_position_bias_41 = contiguous_26.unsqueeze(0)
        contiguous_26 = None
        x_163 = torch._C._nn.pad(layer_norm_29, (0, 0, 0, 0, 0, 0), "constant", None)
        layer_norm_29 = None
        x_164 = torch.roll(x_163, shifts=(-3, -3), dims=(1, 2))
        x_163 = None
        x_165 = x_164.view(1, 2, 7, 2, 7, 512)
        x_164 = None
        permute_60 = x_165.permute(0, 1, 3, 2, 4, 5)
        x_165 = None
        x_166 = permute_60.reshape(4, 49, 512)
        permute_60 = None
        qkv_26 = torch._C._nn.linear(
            x_166,
            l_self_modules_features_modules_5_modules_9_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_features_modules_5_modules_9_modules_attn_modules_qkv_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_9_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_features_modules_5_modules_9_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_59 = qkv_26.reshape(4, 49, 3, 16, 32)
        qkv_26 = None
        qkv_27 = reshape_59.permute(2, 0, 3, 1, 4)
        reshape_59 = None
        q_26 = qkv_27[0]
        k_13 = qkv_27[1]
        v_13 = qkv_27[2]
        qkv_27 = None
        q_27 = q_26 * 0.1767766952966369
        q_26 = None
        transpose_26 = k_13.transpose(-2, -1)
        k_13 = None
        attn_70 = q_27.matmul(transpose_26)
        q_27 = transpose_26 = None
        attn_71 = attn_70 + relative_position_bias_41
        attn_70 = relative_position_bias_41 = None
        attn_mask_30 = x_166.new_zeros((14, 14))
        x_166 = None
        attn_mask_30[(slice(0, -7, None), slice(0, -7, None))] = 0
        setitem_54 = attn_mask_30
        setitem_54 = None
        attn_mask_30[(slice(0, -7, None), slice(-7, -3, None))] = 1
        setitem_55 = attn_mask_30
        setitem_55 = None
        attn_mask_30[(slice(0, -7, None), slice(-3, None, None))] = 2
        setitem_56 = attn_mask_30
        setitem_56 = None
        attn_mask_30[(slice(-7, -3, None), slice(0, -7, None))] = 3
        setitem_57 = attn_mask_30
        setitem_57 = None
        attn_mask_30[(slice(-7, -3, None), slice(-7, -3, None))] = 4
        setitem_58 = attn_mask_30
        setitem_58 = None
        attn_mask_30[(slice(-7, -3, None), slice(-3, None, None))] = 5
        setitem_59 = attn_mask_30
        setitem_59 = None
        attn_mask_30[(slice(-3, None, None), slice(0, -7, None))] = 6
        setitem_60 = attn_mask_30
        setitem_60 = None
        attn_mask_30[(slice(-3, None, None), slice(-7, -3, None))] = 7
        setitem_61 = attn_mask_30
        setitem_61 = None
        attn_mask_30[(slice(-3, None, None), slice(-3, None, None))] = 8
        setitem_62 = attn_mask_30
        setitem_62 = None
        attn_mask_31 = attn_mask_30.view(2, 7, 2, 7)
        attn_mask_30 = None
        permute_62 = attn_mask_31.permute(0, 2, 1, 3)
        attn_mask_31 = None
        attn_mask_32 = permute_62.reshape(4, 49)
        permute_62 = None
        unsqueeze_38 = attn_mask_32.unsqueeze(1)
        unsqueeze_39 = attn_mask_32.unsqueeze(2)
        attn_mask_32 = None
        attn_mask_33 = unsqueeze_38 - unsqueeze_39
        unsqueeze_38 = unsqueeze_39 = None
        ne_6 = attn_mask_33 != 0
        masked_fill_12 = attn_mask_33.masked_fill(ne_6, -100.0)
        ne_6 = None
        eq_6 = attn_mask_33 == 0
        attn_mask_33 = None
        attn_mask_34 = masked_fill_12.masked_fill(eq_6, 0.0)
        masked_fill_12 = eq_6 = None
        attn_72 = attn_71.view(1, 4, 16, 49, 49)
        attn_71 = None
        unsqueeze_40 = attn_mask_34.unsqueeze(1)
        attn_mask_34 = None
        unsqueeze_41 = unsqueeze_40.unsqueeze(0)
        unsqueeze_40 = None
        attn_73 = attn_72 + unsqueeze_41
        attn_72 = unsqueeze_41 = None
        attn_74 = attn_73.view(-1, 16, 49, 49)
        attn_73 = None
        attn_75 = torch.nn.functional.softmax(attn_74, dim=-1)
        attn_74 = None
        attn_76 = torch.nn.functional.dropout(attn_75, p=0.0, training=False)
        attn_75 = None
        matmul_27 = attn_76.matmul(v_13)
        attn_76 = v_13 = None
        transpose_27 = matmul_27.transpose(1, 2)
        matmul_27 = None
        x_167 = transpose_27.reshape(4, 49, 512)
        transpose_27 = None
        x_168 = torch._C._nn.linear(
            x_167,
            l_self_modules_features_modules_5_modules_9_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_5_modules_9_modules_attn_modules_proj_parameters_bias_,
        )
        x_167 = l_self_modules_features_modules_5_modules_9_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_5_modules_9_modules_attn_modules_proj_parameters_bias_ = (None)
        x_169 = torch.nn.functional.dropout(x_168, p=0.0, training=False)
        x_168 = None
        x_170 = x_169.view(1, 2, 2, 7, 7, 512)
        x_169 = None
        permute_63 = x_170.permute(0, 1, 3, 2, 4, 5)
        x_170 = None
        x_171 = permute_63.reshape(1, 14, 14, 512)
        permute_63 = None
        x_172 = torch.roll(x_171, shifts=(3, 3), dims=(1, 2))
        x_171 = None
        getitem_77 = x_172[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_172 = None
        x_173 = getitem_77.contiguous()
        getitem_77 = None
        _log_api_usage_once_26 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_26 = None
        x_174 = x_162 + x_173
        x_162 = x_173 = None
        layer_norm_30 = torch.nn.functional.layer_norm(
            x_174,
            (512,),
            l_self_modules_features_modules_5_modules_9_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_5_modules_9_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_5_modules_9_modules_norm2_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_9_modules_norm2_parameters_bias_
        ) = None
        input_69 = torch._C._nn.linear(
            layer_norm_30,
            l_self_modules_features_modules_5_modules_9_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_9_modules_mlp_modules_0_parameters_bias_,
        )
        layer_norm_30 = l_self_modules_features_modules_5_modules_9_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_9_modules_mlp_modules_0_parameters_bias_ = (None)
        input_70 = torch._C._nn.gelu(input_69, approximate="none")
        input_69 = None
        input_71 = torch.nn.functional.dropout(input_70, 0.0, False, False)
        input_70 = None
        input_72 = torch._C._nn.linear(
            input_71,
            l_self_modules_features_modules_5_modules_9_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_9_modules_mlp_modules_3_parameters_bias_,
        )
        input_71 = l_self_modules_features_modules_5_modules_9_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_9_modules_mlp_modules_3_parameters_bias_ = (None)
        input_73 = torch.nn.functional.dropout(input_72, 0.0, False, False)
        input_72 = None
        _log_api_usage_once_27 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_27 = None
        x_175 = x_174 + input_73
        x_174 = input_73 = None
        layer_norm_31 = torch.nn.functional.layer_norm(
            x_175,
            (512,),
            l_self_modules_features_modules_5_modules_10_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_5_modules_10_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_5_modules_10_modules_norm1_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_10_modules_norm1_parameters_bias_
        ) = None
        relative_position_bias_42 = l_self_modules_features_modules_5_modules_10_modules_attn_parameters_relative_position_bias_table_[
            l_self_modules_features_modules_5_modules_10_modules_attn_buffers_relative_position_index_
        ]
        l_self_modules_features_modules_5_modules_10_modules_attn_parameters_relative_position_bias_table_ = l_self_modules_features_modules_5_modules_10_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_43 = relative_position_bias_42.view(49, 49, -1)
        relative_position_bias_42 = None
        permute_64 = relative_position_bias_43.permute(2, 0, 1)
        relative_position_bias_43 = None
        contiguous_28 = permute_64.contiguous()
        permute_64 = None
        relative_position_bias_44 = contiguous_28.unsqueeze(0)
        contiguous_28 = None
        x_176 = torch._C._nn.pad(layer_norm_31, (0, 0, 0, 0, 0, 0), "constant", None)
        layer_norm_31 = None
        x_177 = x_176.view(1, 2, 7, 2, 7, 512)
        x_176 = None
        permute_65 = x_177.permute(0, 1, 3, 2, 4, 5)
        x_177 = None
        x_178 = permute_65.reshape(4, 49, 512)
        permute_65 = None
        qkv_28 = torch._C._nn.linear(
            x_178,
            l_self_modules_features_modules_5_modules_10_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_features_modules_5_modules_10_modules_attn_modules_qkv_parameters_bias_,
        )
        x_178 = l_self_modules_features_modules_5_modules_10_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_features_modules_5_modules_10_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_64 = qkv_28.reshape(4, 49, 3, 16, 32)
        qkv_28 = None
        qkv_29 = reshape_64.permute(2, 0, 3, 1, 4)
        reshape_64 = None
        q_28 = qkv_29[0]
        k_14 = qkv_29[1]
        v_14 = qkv_29[2]
        qkv_29 = None
        q_29 = q_28 * 0.1767766952966369
        q_28 = None
        transpose_28 = k_14.transpose(-2, -1)
        k_14 = None
        attn_77 = q_29.matmul(transpose_28)
        q_29 = transpose_28 = None
        attn_78 = attn_77 + relative_position_bias_44
        attn_77 = relative_position_bias_44 = None
        attn_79 = torch.nn.functional.softmax(attn_78, dim=-1)
        attn_78 = None
        attn_80 = torch.nn.functional.dropout(attn_79, p=0.0, training=False)
        attn_79 = None
        matmul_29 = attn_80.matmul(v_14)
        attn_80 = v_14 = None
        transpose_29 = matmul_29.transpose(1, 2)
        matmul_29 = None
        x_179 = transpose_29.reshape(4, 49, 512)
        transpose_29 = None
        x_180 = torch._C._nn.linear(
            x_179,
            l_self_modules_features_modules_5_modules_10_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_5_modules_10_modules_attn_modules_proj_parameters_bias_,
        )
        x_179 = l_self_modules_features_modules_5_modules_10_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_5_modules_10_modules_attn_modules_proj_parameters_bias_ = (None)
        x_181 = torch.nn.functional.dropout(x_180, p=0.0, training=False)
        x_180 = None
        x_182 = x_181.view(1, 2, 2, 7, 7, 512)
        x_181 = None
        permute_67 = x_182.permute(0, 1, 3, 2, 4, 5)
        x_182 = None
        x_183 = permute_67.reshape(1, 14, 14, 512)
        permute_67 = None
        getitem_82 = x_183[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_183 = None
        x_184 = getitem_82.contiguous()
        getitem_82 = None
        _log_api_usage_once_28 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_28 = None
        x_185 = x_175 + x_184
        x_175 = x_184 = None
        layer_norm_32 = torch.nn.functional.layer_norm(
            x_185,
            (512,),
            l_self_modules_features_modules_5_modules_10_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_5_modules_10_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_5_modules_10_modules_norm2_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_10_modules_norm2_parameters_bias_
        ) = None
        input_74 = torch._C._nn.linear(
            layer_norm_32,
            l_self_modules_features_modules_5_modules_10_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_10_modules_mlp_modules_0_parameters_bias_,
        )
        layer_norm_32 = l_self_modules_features_modules_5_modules_10_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_10_modules_mlp_modules_0_parameters_bias_ = (None)
        input_75 = torch._C._nn.gelu(input_74, approximate="none")
        input_74 = None
        input_76 = torch.nn.functional.dropout(input_75, 0.0, False, False)
        input_75 = None
        input_77 = torch._C._nn.linear(
            input_76,
            l_self_modules_features_modules_5_modules_10_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_10_modules_mlp_modules_3_parameters_bias_,
        )
        input_76 = l_self_modules_features_modules_5_modules_10_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_10_modules_mlp_modules_3_parameters_bias_ = (None)
        input_78 = torch.nn.functional.dropout(input_77, 0.0, False, False)
        input_77 = None
        _log_api_usage_once_29 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_29 = None
        x_186 = x_185 + input_78
        x_185 = input_78 = None
        layer_norm_33 = torch.nn.functional.layer_norm(
            x_186,
            (512,),
            l_self_modules_features_modules_5_modules_11_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_5_modules_11_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_5_modules_11_modules_norm1_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_11_modules_norm1_parameters_bias_
        ) = None
        relative_position_bias_45 = l_self_modules_features_modules_5_modules_11_modules_attn_parameters_relative_position_bias_table_[
            l_self_modules_features_modules_5_modules_11_modules_attn_buffers_relative_position_index_
        ]
        l_self_modules_features_modules_5_modules_11_modules_attn_parameters_relative_position_bias_table_ = l_self_modules_features_modules_5_modules_11_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_46 = relative_position_bias_45.view(49, 49, -1)
        relative_position_bias_45 = None
        permute_68 = relative_position_bias_46.permute(2, 0, 1)
        relative_position_bias_46 = None
        contiguous_30 = permute_68.contiguous()
        permute_68 = None
        relative_position_bias_47 = contiguous_30.unsqueeze(0)
        contiguous_30 = None
        x_187 = torch._C._nn.pad(layer_norm_33, (0, 0, 0, 0, 0, 0), "constant", None)
        layer_norm_33 = None
        x_188 = torch.roll(x_187, shifts=(-3, -3), dims=(1, 2))
        x_187 = None
        x_189 = x_188.view(1, 2, 7, 2, 7, 512)
        x_188 = None
        permute_69 = x_189.permute(0, 1, 3, 2, 4, 5)
        x_189 = None
        x_190 = permute_69.reshape(4, 49, 512)
        permute_69 = None
        qkv_30 = torch._C._nn.linear(
            x_190,
            l_self_modules_features_modules_5_modules_11_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_features_modules_5_modules_11_modules_attn_modules_qkv_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_11_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_features_modules_5_modules_11_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_68 = qkv_30.reshape(4, 49, 3, 16, 32)
        qkv_30 = None
        qkv_31 = reshape_68.permute(2, 0, 3, 1, 4)
        reshape_68 = None
        q_30 = qkv_31[0]
        k_15 = qkv_31[1]
        v_15 = qkv_31[2]
        qkv_31 = None
        q_31 = q_30 * 0.1767766952966369
        q_30 = None
        transpose_30 = k_15.transpose(-2, -1)
        k_15 = None
        attn_81 = q_31.matmul(transpose_30)
        q_31 = transpose_30 = None
        attn_82 = attn_81 + relative_position_bias_47
        attn_81 = relative_position_bias_47 = None
        attn_mask_35 = x_190.new_zeros((14, 14))
        x_190 = None
        attn_mask_35[(slice(0, -7, None), slice(0, -7, None))] = 0
        setitem_63 = attn_mask_35
        setitem_63 = None
        attn_mask_35[(slice(0, -7, None), slice(-7, -3, None))] = 1
        setitem_64 = attn_mask_35
        setitem_64 = None
        attn_mask_35[(slice(0, -7, None), slice(-3, None, None))] = 2
        setitem_65 = attn_mask_35
        setitem_65 = None
        attn_mask_35[(slice(-7, -3, None), slice(0, -7, None))] = 3
        setitem_66 = attn_mask_35
        setitem_66 = None
        attn_mask_35[(slice(-7, -3, None), slice(-7, -3, None))] = 4
        setitem_67 = attn_mask_35
        setitem_67 = None
        attn_mask_35[(slice(-7, -3, None), slice(-3, None, None))] = 5
        setitem_68 = attn_mask_35
        setitem_68 = None
        attn_mask_35[(slice(-3, None, None), slice(0, -7, None))] = 6
        setitem_69 = attn_mask_35
        setitem_69 = None
        attn_mask_35[(slice(-3, None, None), slice(-7, -3, None))] = 7
        setitem_70 = attn_mask_35
        setitem_70 = None
        attn_mask_35[(slice(-3, None, None), slice(-3, None, None))] = 8
        setitem_71 = attn_mask_35
        setitem_71 = None
        attn_mask_36 = attn_mask_35.view(2, 7, 2, 7)
        attn_mask_35 = None
        permute_71 = attn_mask_36.permute(0, 2, 1, 3)
        attn_mask_36 = None
        attn_mask_37 = permute_71.reshape(4, 49)
        permute_71 = None
        unsqueeze_44 = attn_mask_37.unsqueeze(1)
        unsqueeze_45 = attn_mask_37.unsqueeze(2)
        attn_mask_37 = None
        attn_mask_38 = unsqueeze_44 - unsqueeze_45
        unsqueeze_44 = unsqueeze_45 = None
        ne_7 = attn_mask_38 != 0
        masked_fill_14 = attn_mask_38.masked_fill(ne_7, -100.0)
        ne_7 = None
        eq_7 = attn_mask_38 == 0
        attn_mask_38 = None
        attn_mask_39 = masked_fill_14.masked_fill(eq_7, 0.0)
        masked_fill_14 = eq_7 = None
        attn_83 = attn_82.view(1, 4, 16, 49, 49)
        attn_82 = None
        unsqueeze_46 = attn_mask_39.unsqueeze(1)
        attn_mask_39 = None
        unsqueeze_47 = unsqueeze_46.unsqueeze(0)
        unsqueeze_46 = None
        attn_84 = attn_83 + unsqueeze_47
        attn_83 = unsqueeze_47 = None
        attn_85 = attn_84.view(-1, 16, 49, 49)
        attn_84 = None
        attn_86 = torch.nn.functional.softmax(attn_85, dim=-1)
        attn_85 = None
        attn_87 = torch.nn.functional.dropout(attn_86, p=0.0, training=False)
        attn_86 = None
        matmul_31 = attn_87.matmul(v_15)
        attn_87 = v_15 = None
        transpose_31 = matmul_31.transpose(1, 2)
        matmul_31 = None
        x_191 = transpose_31.reshape(4, 49, 512)
        transpose_31 = None
        x_192 = torch._C._nn.linear(
            x_191,
            l_self_modules_features_modules_5_modules_11_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_5_modules_11_modules_attn_modules_proj_parameters_bias_,
        )
        x_191 = l_self_modules_features_modules_5_modules_11_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_5_modules_11_modules_attn_modules_proj_parameters_bias_ = (None)
        x_193 = torch.nn.functional.dropout(x_192, p=0.0, training=False)
        x_192 = None
        x_194 = x_193.view(1, 2, 2, 7, 7, 512)
        x_193 = None
        permute_72 = x_194.permute(0, 1, 3, 2, 4, 5)
        x_194 = None
        x_195 = permute_72.reshape(1, 14, 14, 512)
        permute_72 = None
        x_196 = torch.roll(x_195, shifts=(3, 3), dims=(1, 2))
        x_195 = None
        getitem_87 = x_196[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_196 = None
        x_197 = getitem_87.contiguous()
        getitem_87 = None
        _log_api_usage_once_30 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_30 = None
        x_198 = x_186 + x_197
        x_186 = x_197 = None
        layer_norm_34 = torch.nn.functional.layer_norm(
            x_198,
            (512,),
            l_self_modules_features_modules_5_modules_11_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_5_modules_11_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_5_modules_11_modules_norm2_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_11_modules_norm2_parameters_bias_
        ) = None
        input_79 = torch._C._nn.linear(
            layer_norm_34,
            l_self_modules_features_modules_5_modules_11_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_11_modules_mlp_modules_0_parameters_bias_,
        )
        layer_norm_34 = l_self_modules_features_modules_5_modules_11_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_11_modules_mlp_modules_0_parameters_bias_ = (None)
        input_80 = torch._C._nn.gelu(input_79, approximate="none")
        input_79 = None
        input_81 = torch.nn.functional.dropout(input_80, 0.0, False, False)
        input_80 = None
        input_82 = torch._C._nn.linear(
            input_81,
            l_self_modules_features_modules_5_modules_11_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_11_modules_mlp_modules_3_parameters_bias_,
        )
        input_81 = l_self_modules_features_modules_5_modules_11_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_11_modules_mlp_modules_3_parameters_bias_ = (None)
        input_83 = torch.nn.functional.dropout(input_82, 0.0, False, False)
        input_82 = None
        _log_api_usage_once_31 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_31 = None
        x_199 = x_198 + input_83
        x_198 = input_83 = None
        layer_norm_35 = torch.nn.functional.layer_norm(
            x_199,
            (512,),
            l_self_modules_features_modules_5_modules_12_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_5_modules_12_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_5_modules_12_modules_norm1_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_12_modules_norm1_parameters_bias_
        ) = None
        relative_position_bias_48 = l_self_modules_features_modules_5_modules_12_modules_attn_parameters_relative_position_bias_table_[
            l_self_modules_features_modules_5_modules_12_modules_attn_buffers_relative_position_index_
        ]
        l_self_modules_features_modules_5_modules_12_modules_attn_parameters_relative_position_bias_table_ = l_self_modules_features_modules_5_modules_12_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_49 = relative_position_bias_48.view(49, 49, -1)
        relative_position_bias_48 = None
        permute_73 = relative_position_bias_49.permute(2, 0, 1)
        relative_position_bias_49 = None
        contiguous_32 = permute_73.contiguous()
        permute_73 = None
        relative_position_bias_50 = contiguous_32.unsqueeze(0)
        contiguous_32 = None
        x_200 = torch._C._nn.pad(layer_norm_35, (0, 0, 0, 0, 0, 0), "constant", None)
        layer_norm_35 = None
        x_201 = x_200.view(1, 2, 7, 2, 7, 512)
        x_200 = None
        permute_74 = x_201.permute(0, 1, 3, 2, 4, 5)
        x_201 = None
        x_202 = permute_74.reshape(4, 49, 512)
        permute_74 = None
        qkv_32 = torch._C._nn.linear(
            x_202,
            l_self_modules_features_modules_5_modules_12_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_features_modules_5_modules_12_modules_attn_modules_qkv_parameters_bias_,
        )
        x_202 = l_self_modules_features_modules_5_modules_12_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_features_modules_5_modules_12_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_73 = qkv_32.reshape(4, 49, 3, 16, 32)
        qkv_32 = None
        qkv_33 = reshape_73.permute(2, 0, 3, 1, 4)
        reshape_73 = None
        q_32 = qkv_33[0]
        k_16 = qkv_33[1]
        v_16 = qkv_33[2]
        qkv_33 = None
        q_33 = q_32 * 0.1767766952966369
        q_32 = None
        transpose_32 = k_16.transpose(-2, -1)
        k_16 = None
        attn_88 = q_33.matmul(transpose_32)
        q_33 = transpose_32 = None
        attn_89 = attn_88 + relative_position_bias_50
        attn_88 = relative_position_bias_50 = None
        attn_90 = torch.nn.functional.softmax(attn_89, dim=-1)
        attn_89 = None
        attn_91 = torch.nn.functional.dropout(attn_90, p=0.0, training=False)
        attn_90 = None
        matmul_33 = attn_91.matmul(v_16)
        attn_91 = v_16 = None
        transpose_33 = matmul_33.transpose(1, 2)
        matmul_33 = None
        x_203 = transpose_33.reshape(4, 49, 512)
        transpose_33 = None
        x_204 = torch._C._nn.linear(
            x_203,
            l_self_modules_features_modules_5_modules_12_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_5_modules_12_modules_attn_modules_proj_parameters_bias_,
        )
        x_203 = l_self_modules_features_modules_5_modules_12_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_5_modules_12_modules_attn_modules_proj_parameters_bias_ = (None)
        x_205 = torch.nn.functional.dropout(x_204, p=0.0, training=False)
        x_204 = None
        x_206 = x_205.view(1, 2, 2, 7, 7, 512)
        x_205 = None
        permute_76 = x_206.permute(0, 1, 3, 2, 4, 5)
        x_206 = None
        x_207 = permute_76.reshape(1, 14, 14, 512)
        permute_76 = None
        getitem_92 = x_207[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_207 = None
        x_208 = getitem_92.contiguous()
        getitem_92 = None
        _log_api_usage_once_32 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_32 = None
        x_209 = x_199 + x_208
        x_199 = x_208 = None
        layer_norm_36 = torch.nn.functional.layer_norm(
            x_209,
            (512,),
            l_self_modules_features_modules_5_modules_12_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_5_modules_12_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_5_modules_12_modules_norm2_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_12_modules_norm2_parameters_bias_
        ) = None
        input_84 = torch._C._nn.linear(
            layer_norm_36,
            l_self_modules_features_modules_5_modules_12_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_12_modules_mlp_modules_0_parameters_bias_,
        )
        layer_norm_36 = l_self_modules_features_modules_5_modules_12_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_12_modules_mlp_modules_0_parameters_bias_ = (None)
        input_85 = torch._C._nn.gelu(input_84, approximate="none")
        input_84 = None
        input_86 = torch.nn.functional.dropout(input_85, 0.0, False, False)
        input_85 = None
        input_87 = torch._C._nn.linear(
            input_86,
            l_self_modules_features_modules_5_modules_12_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_12_modules_mlp_modules_3_parameters_bias_,
        )
        input_86 = l_self_modules_features_modules_5_modules_12_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_12_modules_mlp_modules_3_parameters_bias_ = (None)
        input_88 = torch.nn.functional.dropout(input_87, 0.0, False, False)
        input_87 = None
        _log_api_usage_once_33 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_33 = None
        x_210 = x_209 + input_88
        x_209 = input_88 = None
        layer_norm_37 = torch.nn.functional.layer_norm(
            x_210,
            (512,),
            l_self_modules_features_modules_5_modules_13_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_5_modules_13_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_5_modules_13_modules_norm1_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_13_modules_norm1_parameters_bias_
        ) = None
        relative_position_bias_51 = l_self_modules_features_modules_5_modules_13_modules_attn_parameters_relative_position_bias_table_[
            l_self_modules_features_modules_5_modules_13_modules_attn_buffers_relative_position_index_
        ]
        l_self_modules_features_modules_5_modules_13_modules_attn_parameters_relative_position_bias_table_ = l_self_modules_features_modules_5_modules_13_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_52 = relative_position_bias_51.view(49, 49, -1)
        relative_position_bias_51 = None
        permute_77 = relative_position_bias_52.permute(2, 0, 1)
        relative_position_bias_52 = None
        contiguous_34 = permute_77.contiguous()
        permute_77 = None
        relative_position_bias_53 = contiguous_34.unsqueeze(0)
        contiguous_34 = None
        x_211 = torch._C._nn.pad(layer_norm_37, (0, 0, 0, 0, 0, 0), "constant", None)
        layer_norm_37 = None
        x_212 = torch.roll(x_211, shifts=(-3, -3), dims=(1, 2))
        x_211 = None
        x_213 = x_212.view(1, 2, 7, 2, 7, 512)
        x_212 = None
        permute_78 = x_213.permute(0, 1, 3, 2, 4, 5)
        x_213 = None
        x_214 = permute_78.reshape(4, 49, 512)
        permute_78 = None
        qkv_34 = torch._C._nn.linear(
            x_214,
            l_self_modules_features_modules_5_modules_13_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_features_modules_5_modules_13_modules_attn_modules_qkv_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_13_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_features_modules_5_modules_13_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_77 = qkv_34.reshape(4, 49, 3, 16, 32)
        qkv_34 = None
        qkv_35 = reshape_77.permute(2, 0, 3, 1, 4)
        reshape_77 = None
        q_34 = qkv_35[0]
        k_17 = qkv_35[1]
        v_17 = qkv_35[2]
        qkv_35 = None
        q_35 = q_34 * 0.1767766952966369
        q_34 = None
        transpose_34 = k_17.transpose(-2, -1)
        k_17 = None
        attn_92 = q_35.matmul(transpose_34)
        q_35 = transpose_34 = None
        attn_93 = attn_92 + relative_position_bias_53
        attn_92 = relative_position_bias_53 = None
        attn_mask_40 = x_214.new_zeros((14, 14))
        x_214 = None
        attn_mask_40[(slice(0, -7, None), slice(0, -7, None))] = 0
        setitem_72 = attn_mask_40
        setitem_72 = None
        attn_mask_40[(slice(0, -7, None), slice(-7, -3, None))] = 1
        setitem_73 = attn_mask_40
        setitem_73 = None
        attn_mask_40[(slice(0, -7, None), slice(-3, None, None))] = 2
        setitem_74 = attn_mask_40
        setitem_74 = None
        attn_mask_40[(slice(-7, -3, None), slice(0, -7, None))] = 3
        setitem_75 = attn_mask_40
        setitem_75 = None
        attn_mask_40[(slice(-7, -3, None), slice(-7, -3, None))] = 4
        setitem_76 = attn_mask_40
        setitem_76 = None
        attn_mask_40[(slice(-7, -3, None), slice(-3, None, None))] = 5
        setitem_77 = attn_mask_40
        setitem_77 = None
        attn_mask_40[(slice(-3, None, None), slice(0, -7, None))] = 6
        setitem_78 = attn_mask_40
        setitem_78 = None
        attn_mask_40[(slice(-3, None, None), slice(-7, -3, None))] = 7
        setitem_79 = attn_mask_40
        setitem_79 = None
        attn_mask_40[(slice(-3, None, None), slice(-3, None, None))] = 8
        setitem_80 = attn_mask_40
        setitem_80 = None
        attn_mask_41 = attn_mask_40.view(2, 7, 2, 7)
        attn_mask_40 = None
        permute_80 = attn_mask_41.permute(0, 2, 1, 3)
        attn_mask_41 = None
        attn_mask_42 = permute_80.reshape(4, 49)
        permute_80 = None
        unsqueeze_50 = attn_mask_42.unsqueeze(1)
        unsqueeze_51 = attn_mask_42.unsqueeze(2)
        attn_mask_42 = None
        attn_mask_43 = unsqueeze_50 - unsqueeze_51
        unsqueeze_50 = unsqueeze_51 = None
        ne_8 = attn_mask_43 != 0
        masked_fill_16 = attn_mask_43.masked_fill(ne_8, -100.0)
        ne_8 = None
        eq_8 = attn_mask_43 == 0
        attn_mask_43 = None
        attn_mask_44 = masked_fill_16.masked_fill(eq_8, 0.0)
        masked_fill_16 = eq_8 = None
        attn_94 = attn_93.view(1, 4, 16, 49, 49)
        attn_93 = None
        unsqueeze_52 = attn_mask_44.unsqueeze(1)
        attn_mask_44 = None
        unsqueeze_53 = unsqueeze_52.unsqueeze(0)
        unsqueeze_52 = None
        attn_95 = attn_94 + unsqueeze_53
        attn_94 = unsqueeze_53 = None
        attn_96 = attn_95.view(-1, 16, 49, 49)
        attn_95 = None
        attn_97 = torch.nn.functional.softmax(attn_96, dim=-1)
        attn_96 = None
        attn_98 = torch.nn.functional.dropout(attn_97, p=0.0, training=False)
        attn_97 = None
        matmul_35 = attn_98.matmul(v_17)
        attn_98 = v_17 = None
        transpose_35 = matmul_35.transpose(1, 2)
        matmul_35 = None
        x_215 = transpose_35.reshape(4, 49, 512)
        transpose_35 = None
        x_216 = torch._C._nn.linear(
            x_215,
            l_self_modules_features_modules_5_modules_13_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_5_modules_13_modules_attn_modules_proj_parameters_bias_,
        )
        x_215 = l_self_modules_features_modules_5_modules_13_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_5_modules_13_modules_attn_modules_proj_parameters_bias_ = (None)
        x_217 = torch.nn.functional.dropout(x_216, p=0.0, training=False)
        x_216 = None
        x_218 = x_217.view(1, 2, 2, 7, 7, 512)
        x_217 = None
        permute_81 = x_218.permute(0, 1, 3, 2, 4, 5)
        x_218 = None
        x_219 = permute_81.reshape(1, 14, 14, 512)
        permute_81 = None
        x_220 = torch.roll(x_219, shifts=(3, 3), dims=(1, 2))
        x_219 = None
        getitem_97 = x_220[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_220 = None
        x_221 = getitem_97.contiguous()
        getitem_97 = None
        _log_api_usage_once_34 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_34 = None
        x_222 = x_210 + x_221
        x_210 = x_221 = None
        layer_norm_38 = torch.nn.functional.layer_norm(
            x_222,
            (512,),
            l_self_modules_features_modules_5_modules_13_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_5_modules_13_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_5_modules_13_modules_norm2_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_13_modules_norm2_parameters_bias_
        ) = None
        input_89 = torch._C._nn.linear(
            layer_norm_38,
            l_self_modules_features_modules_5_modules_13_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_13_modules_mlp_modules_0_parameters_bias_,
        )
        layer_norm_38 = l_self_modules_features_modules_5_modules_13_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_13_modules_mlp_modules_0_parameters_bias_ = (None)
        input_90 = torch._C._nn.gelu(input_89, approximate="none")
        input_89 = None
        input_91 = torch.nn.functional.dropout(input_90, 0.0, False, False)
        input_90 = None
        input_92 = torch._C._nn.linear(
            input_91,
            l_self_modules_features_modules_5_modules_13_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_13_modules_mlp_modules_3_parameters_bias_,
        )
        input_91 = l_self_modules_features_modules_5_modules_13_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_13_modules_mlp_modules_3_parameters_bias_ = (None)
        input_93 = torch.nn.functional.dropout(input_92, 0.0, False, False)
        input_92 = None
        _log_api_usage_once_35 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_35 = None
        x_223 = x_222 + input_93
        x_222 = input_93 = None
        layer_norm_39 = torch.nn.functional.layer_norm(
            x_223,
            (512,),
            l_self_modules_features_modules_5_modules_14_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_5_modules_14_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_5_modules_14_modules_norm1_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_14_modules_norm1_parameters_bias_
        ) = None
        relative_position_bias_54 = l_self_modules_features_modules_5_modules_14_modules_attn_parameters_relative_position_bias_table_[
            l_self_modules_features_modules_5_modules_14_modules_attn_buffers_relative_position_index_
        ]
        l_self_modules_features_modules_5_modules_14_modules_attn_parameters_relative_position_bias_table_ = l_self_modules_features_modules_5_modules_14_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_55 = relative_position_bias_54.view(49, 49, -1)
        relative_position_bias_54 = None
        permute_82 = relative_position_bias_55.permute(2, 0, 1)
        relative_position_bias_55 = None
        contiguous_36 = permute_82.contiguous()
        permute_82 = None
        relative_position_bias_56 = contiguous_36.unsqueeze(0)
        contiguous_36 = None
        x_224 = torch._C._nn.pad(layer_norm_39, (0, 0, 0, 0, 0, 0), "constant", None)
        layer_norm_39 = None
        x_225 = x_224.view(1, 2, 7, 2, 7, 512)
        x_224 = None
        permute_83 = x_225.permute(0, 1, 3, 2, 4, 5)
        x_225 = None
        x_226 = permute_83.reshape(4, 49, 512)
        permute_83 = None
        qkv_36 = torch._C._nn.linear(
            x_226,
            l_self_modules_features_modules_5_modules_14_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_features_modules_5_modules_14_modules_attn_modules_qkv_parameters_bias_,
        )
        x_226 = l_self_modules_features_modules_5_modules_14_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_features_modules_5_modules_14_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_82 = qkv_36.reshape(4, 49, 3, 16, 32)
        qkv_36 = None
        qkv_37 = reshape_82.permute(2, 0, 3, 1, 4)
        reshape_82 = None
        q_36 = qkv_37[0]
        k_18 = qkv_37[1]
        v_18 = qkv_37[2]
        qkv_37 = None
        q_37 = q_36 * 0.1767766952966369
        q_36 = None
        transpose_36 = k_18.transpose(-2, -1)
        k_18 = None
        attn_99 = q_37.matmul(transpose_36)
        q_37 = transpose_36 = None
        attn_100 = attn_99 + relative_position_bias_56
        attn_99 = relative_position_bias_56 = None
        attn_101 = torch.nn.functional.softmax(attn_100, dim=-1)
        attn_100 = None
        attn_102 = torch.nn.functional.dropout(attn_101, p=0.0, training=False)
        attn_101 = None
        matmul_37 = attn_102.matmul(v_18)
        attn_102 = v_18 = None
        transpose_37 = matmul_37.transpose(1, 2)
        matmul_37 = None
        x_227 = transpose_37.reshape(4, 49, 512)
        transpose_37 = None
        x_228 = torch._C._nn.linear(
            x_227,
            l_self_modules_features_modules_5_modules_14_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_5_modules_14_modules_attn_modules_proj_parameters_bias_,
        )
        x_227 = l_self_modules_features_modules_5_modules_14_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_5_modules_14_modules_attn_modules_proj_parameters_bias_ = (None)
        x_229 = torch.nn.functional.dropout(x_228, p=0.0, training=False)
        x_228 = None
        x_230 = x_229.view(1, 2, 2, 7, 7, 512)
        x_229 = None
        permute_85 = x_230.permute(0, 1, 3, 2, 4, 5)
        x_230 = None
        x_231 = permute_85.reshape(1, 14, 14, 512)
        permute_85 = None
        getitem_102 = x_231[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_231 = None
        x_232 = getitem_102.contiguous()
        getitem_102 = None
        _log_api_usage_once_36 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_36 = None
        x_233 = x_223 + x_232
        x_223 = x_232 = None
        layer_norm_40 = torch.nn.functional.layer_norm(
            x_233,
            (512,),
            l_self_modules_features_modules_5_modules_14_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_5_modules_14_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_5_modules_14_modules_norm2_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_14_modules_norm2_parameters_bias_
        ) = None
        input_94 = torch._C._nn.linear(
            layer_norm_40,
            l_self_modules_features_modules_5_modules_14_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_14_modules_mlp_modules_0_parameters_bias_,
        )
        layer_norm_40 = l_self_modules_features_modules_5_modules_14_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_14_modules_mlp_modules_0_parameters_bias_ = (None)
        input_95 = torch._C._nn.gelu(input_94, approximate="none")
        input_94 = None
        input_96 = torch.nn.functional.dropout(input_95, 0.0, False, False)
        input_95 = None
        input_97 = torch._C._nn.linear(
            input_96,
            l_self_modules_features_modules_5_modules_14_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_14_modules_mlp_modules_3_parameters_bias_,
        )
        input_96 = l_self_modules_features_modules_5_modules_14_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_14_modules_mlp_modules_3_parameters_bias_ = (None)
        input_98 = torch.nn.functional.dropout(input_97, 0.0, False, False)
        input_97 = None
        _log_api_usage_once_37 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_37 = None
        x_234 = x_233 + input_98
        x_233 = input_98 = None
        layer_norm_41 = torch.nn.functional.layer_norm(
            x_234,
            (512,),
            l_self_modules_features_modules_5_modules_15_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_5_modules_15_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_5_modules_15_modules_norm1_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_15_modules_norm1_parameters_bias_
        ) = None
        relative_position_bias_57 = l_self_modules_features_modules_5_modules_15_modules_attn_parameters_relative_position_bias_table_[
            l_self_modules_features_modules_5_modules_15_modules_attn_buffers_relative_position_index_
        ]
        l_self_modules_features_modules_5_modules_15_modules_attn_parameters_relative_position_bias_table_ = l_self_modules_features_modules_5_modules_15_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_58 = relative_position_bias_57.view(49, 49, -1)
        relative_position_bias_57 = None
        permute_86 = relative_position_bias_58.permute(2, 0, 1)
        relative_position_bias_58 = None
        contiguous_38 = permute_86.contiguous()
        permute_86 = None
        relative_position_bias_59 = contiguous_38.unsqueeze(0)
        contiguous_38 = None
        x_235 = torch._C._nn.pad(layer_norm_41, (0, 0, 0, 0, 0, 0), "constant", None)
        layer_norm_41 = None
        x_236 = torch.roll(x_235, shifts=(-3, -3), dims=(1, 2))
        x_235 = None
        x_237 = x_236.view(1, 2, 7, 2, 7, 512)
        x_236 = None
        permute_87 = x_237.permute(0, 1, 3, 2, 4, 5)
        x_237 = None
        x_238 = permute_87.reshape(4, 49, 512)
        permute_87 = None
        qkv_38 = torch._C._nn.linear(
            x_238,
            l_self_modules_features_modules_5_modules_15_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_features_modules_5_modules_15_modules_attn_modules_qkv_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_15_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_features_modules_5_modules_15_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_86 = qkv_38.reshape(4, 49, 3, 16, 32)
        qkv_38 = None
        qkv_39 = reshape_86.permute(2, 0, 3, 1, 4)
        reshape_86 = None
        q_38 = qkv_39[0]
        k_19 = qkv_39[1]
        v_19 = qkv_39[2]
        qkv_39 = None
        q_39 = q_38 * 0.1767766952966369
        q_38 = None
        transpose_38 = k_19.transpose(-2, -1)
        k_19 = None
        attn_103 = q_39.matmul(transpose_38)
        q_39 = transpose_38 = None
        attn_104 = attn_103 + relative_position_bias_59
        attn_103 = relative_position_bias_59 = None
        attn_mask_45 = x_238.new_zeros((14, 14))
        x_238 = None
        attn_mask_45[(slice(0, -7, None), slice(0, -7, None))] = 0
        setitem_81 = attn_mask_45
        setitem_81 = None
        attn_mask_45[(slice(0, -7, None), slice(-7, -3, None))] = 1
        setitem_82 = attn_mask_45
        setitem_82 = None
        attn_mask_45[(slice(0, -7, None), slice(-3, None, None))] = 2
        setitem_83 = attn_mask_45
        setitem_83 = None
        attn_mask_45[(slice(-7, -3, None), slice(0, -7, None))] = 3
        setitem_84 = attn_mask_45
        setitem_84 = None
        attn_mask_45[(slice(-7, -3, None), slice(-7, -3, None))] = 4
        setitem_85 = attn_mask_45
        setitem_85 = None
        attn_mask_45[(slice(-7, -3, None), slice(-3, None, None))] = 5
        setitem_86 = attn_mask_45
        setitem_86 = None
        attn_mask_45[(slice(-3, None, None), slice(0, -7, None))] = 6
        setitem_87 = attn_mask_45
        setitem_87 = None
        attn_mask_45[(slice(-3, None, None), slice(-7, -3, None))] = 7
        setitem_88 = attn_mask_45
        setitem_88 = None
        attn_mask_45[(slice(-3, None, None), slice(-3, None, None))] = 8
        setitem_89 = attn_mask_45
        setitem_89 = None
        attn_mask_46 = attn_mask_45.view(2, 7, 2, 7)
        attn_mask_45 = None
        permute_89 = attn_mask_46.permute(0, 2, 1, 3)
        attn_mask_46 = None
        attn_mask_47 = permute_89.reshape(4, 49)
        permute_89 = None
        unsqueeze_56 = attn_mask_47.unsqueeze(1)
        unsqueeze_57 = attn_mask_47.unsqueeze(2)
        attn_mask_47 = None
        attn_mask_48 = unsqueeze_56 - unsqueeze_57
        unsqueeze_56 = unsqueeze_57 = None
        ne_9 = attn_mask_48 != 0
        masked_fill_18 = attn_mask_48.masked_fill(ne_9, -100.0)
        ne_9 = None
        eq_9 = attn_mask_48 == 0
        attn_mask_48 = None
        attn_mask_49 = masked_fill_18.masked_fill(eq_9, 0.0)
        masked_fill_18 = eq_9 = None
        attn_105 = attn_104.view(1, 4, 16, 49, 49)
        attn_104 = None
        unsqueeze_58 = attn_mask_49.unsqueeze(1)
        attn_mask_49 = None
        unsqueeze_59 = unsqueeze_58.unsqueeze(0)
        unsqueeze_58 = None
        attn_106 = attn_105 + unsqueeze_59
        attn_105 = unsqueeze_59 = None
        attn_107 = attn_106.view(-1, 16, 49, 49)
        attn_106 = None
        attn_108 = torch.nn.functional.softmax(attn_107, dim=-1)
        attn_107 = None
        attn_109 = torch.nn.functional.dropout(attn_108, p=0.0, training=False)
        attn_108 = None
        matmul_39 = attn_109.matmul(v_19)
        attn_109 = v_19 = None
        transpose_39 = matmul_39.transpose(1, 2)
        matmul_39 = None
        x_239 = transpose_39.reshape(4, 49, 512)
        transpose_39 = None
        x_240 = torch._C._nn.linear(
            x_239,
            l_self_modules_features_modules_5_modules_15_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_5_modules_15_modules_attn_modules_proj_parameters_bias_,
        )
        x_239 = l_self_modules_features_modules_5_modules_15_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_5_modules_15_modules_attn_modules_proj_parameters_bias_ = (None)
        x_241 = torch.nn.functional.dropout(x_240, p=0.0, training=False)
        x_240 = None
        x_242 = x_241.view(1, 2, 2, 7, 7, 512)
        x_241 = None
        permute_90 = x_242.permute(0, 1, 3, 2, 4, 5)
        x_242 = None
        x_243 = permute_90.reshape(1, 14, 14, 512)
        permute_90 = None
        x_244 = torch.roll(x_243, shifts=(3, 3), dims=(1, 2))
        x_243 = None
        getitem_107 = x_244[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_244 = None
        x_245 = getitem_107.contiguous()
        getitem_107 = None
        _log_api_usage_once_38 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_38 = None
        x_246 = x_234 + x_245
        x_234 = x_245 = None
        layer_norm_42 = torch.nn.functional.layer_norm(
            x_246,
            (512,),
            l_self_modules_features_modules_5_modules_15_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_5_modules_15_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_5_modules_15_modules_norm2_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_15_modules_norm2_parameters_bias_
        ) = None
        input_99 = torch._C._nn.linear(
            layer_norm_42,
            l_self_modules_features_modules_5_modules_15_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_15_modules_mlp_modules_0_parameters_bias_,
        )
        layer_norm_42 = l_self_modules_features_modules_5_modules_15_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_15_modules_mlp_modules_0_parameters_bias_ = (None)
        input_100 = torch._C._nn.gelu(input_99, approximate="none")
        input_99 = None
        input_101 = torch.nn.functional.dropout(input_100, 0.0, False, False)
        input_100 = None
        input_102 = torch._C._nn.linear(
            input_101,
            l_self_modules_features_modules_5_modules_15_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_15_modules_mlp_modules_3_parameters_bias_,
        )
        input_101 = l_self_modules_features_modules_5_modules_15_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_15_modules_mlp_modules_3_parameters_bias_ = (None)
        input_103 = torch.nn.functional.dropout(input_102, 0.0, False, False)
        input_102 = None
        _log_api_usage_once_39 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_39 = None
        x_247 = x_246 + input_103
        x_246 = input_103 = None
        layer_norm_43 = torch.nn.functional.layer_norm(
            x_247,
            (512,),
            l_self_modules_features_modules_5_modules_16_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_5_modules_16_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_5_modules_16_modules_norm1_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_16_modules_norm1_parameters_bias_
        ) = None
        relative_position_bias_60 = l_self_modules_features_modules_5_modules_16_modules_attn_parameters_relative_position_bias_table_[
            l_self_modules_features_modules_5_modules_16_modules_attn_buffers_relative_position_index_
        ]
        l_self_modules_features_modules_5_modules_16_modules_attn_parameters_relative_position_bias_table_ = l_self_modules_features_modules_5_modules_16_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_61 = relative_position_bias_60.view(49, 49, -1)
        relative_position_bias_60 = None
        permute_91 = relative_position_bias_61.permute(2, 0, 1)
        relative_position_bias_61 = None
        contiguous_40 = permute_91.contiguous()
        permute_91 = None
        relative_position_bias_62 = contiguous_40.unsqueeze(0)
        contiguous_40 = None
        x_248 = torch._C._nn.pad(layer_norm_43, (0, 0, 0, 0, 0, 0), "constant", None)
        layer_norm_43 = None
        x_249 = x_248.view(1, 2, 7, 2, 7, 512)
        x_248 = None
        permute_92 = x_249.permute(0, 1, 3, 2, 4, 5)
        x_249 = None
        x_250 = permute_92.reshape(4, 49, 512)
        permute_92 = None
        qkv_40 = torch._C._nn.linear(
            x_250,
            l_self_modules_features_modules_5_modules_16_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_features_modules_5_modules_16_modules_attn_modules_qkv_parameters_bias_,
        )
        x_250 = l_self_modules_features_modules_5_modules_16_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_features_modules_5_modules_16_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_91 = qkv_40.reshape(4, 49, 3, 16, 32)
        qkv_40 = None
        qkv_41 = reshape_91.permute(2, 0, 3, 1, 4)
        reshape_91 = None
        q_40 = qkv_41[0]
        k_20 = qkv_41[1]
        v_20 = qkv_41[2]
        qkv_41 = None
        q_41 = q_40 * 0.1767766952966369
        q_40 = None
        transpose_40 = k_20.transpose(-2, -1)
        k_20 = None
        attn_110 = q_41.matmul(transpose_40)
        q_41 = transpose_40 = None
        attn_111 = attn_110 + relative_position_bias_62
        attn_110 = relative_position_bias_62 = None
        attn_112 = torch.nn.functional.softmax(attn_111, dim=-1)
        attn_111 = None
        attn_113 = torch.nn.functional.dropout(attn_112, p=0.0, training=False)
        attn_112 = None
        matmul_41 = attn_113.matmul(v_20)
        attn_113 = v_20 = None
        transpose_41 = matmul_41.transpose(1, 2)
        matmul_41 = None
        x_251 = transpose_41.reshape(4, 49, 512)
        transpose_41 = None
        x_252 = torch._C._nn.linear(
            x_251,
            l_self_modules_features_modules_5_modules_16_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_5_modules_16_modules_attn_modules_proj_parameters_bias_,
        )
        x_251 = l_self_modules_features_modules_5_modules_16_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_5_modules_16_modules_attn_modules_proj_parameters_bias_ = (None)
        x_253 = torch.nn.functional.dropout(x_252, p=0.0, training=False)
        x_252 = None
        x_254 = x_253.view(1, 2, 2, 7, 7, 512)
        x_253 = None
        permute_94 = x_254.permute(0, 1, 3, 2, 4, 5)
        x_254 = None
        x_255 = permute_94.reshape(1, 14, 14, 512)
        permute_94 = None
        getitem_112 = x_255[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_255 = None
        x_256 = getitem_112.contiguous()
        getitem_112 = None
        _log_api_usage_once_40 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_40 = None
        x_257 = x_247 + x_256
        x_247 = x_256 = None
        layer_norm_44 = torch.nn.functional.layer_norm(
            x_257,
            (512,),
            l_self_modules_features_modules_5_modules_16_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_5_modules_16_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_5_modules_16_modules_norm2_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_16_modules_norm2_parameters_bias_
        ) = None
        input_104 = torch._C._nn.linear(
            layer_norm_44,
            l_self_modules_features_modules_5_modules_16_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_16_modules_mlp_modules_0_parameters_bias_,
        )
        layer_norm_44 = l_self_modules_features_modules_5_modules_16_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_16_modules_mlp_modules_0_parameters_bias_ = (None)
        input_105 = torch._C._nn.gelu(input_104, approximate="none")
        input_104 = None
        input_106 = torch.nn.functional.dropout(input_105, 0.0, False, False)
        input_105 = None
        input_107 = torch._C._nn.linear(
            input_106,
            l_self_modules_features_modules_5_modules_16_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_16_modules_mlp_modules_3_parameters_bias_,
        )
        input_106 = l_self_modules_features_modules_5_modules_16_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_16_modules_mlp_modules_3_parameters_bias_ = (None)
        input_108 = torch.nn.functional.dropout(input_107, 0.0, False, False)
        input_107 = None
        _log_api_usage_once_41 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_41 = None
        x_258 = x_257 + input_108
        x_257 = input_108 = None
        layer_norm_45 = torch.nn.functional.layer_norm(
            x_258,
            (512,),
            l_self_modules_features_modules_5_modules_17_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_5_modules_17_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_5_modules_17_modules_norm1_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_17_modules_norm1_parameters_bias_
        ) = None
        relative_position_bias_63 = l_self_modules_features_modules_5_modules_17_modules_attn_parameters_relative_position_bias_table_[
            l_self_modules_features_modules_5_modules_17_modules_attn_buffers_relative_position_index_
        ]
        l_self_modules_features_modules_5_modules_17_modules_attn_parameters_relative_position_bias_table_ = l_self_modules_features_modules_5_modules_17_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_64 = relative_position_bias_63.view(49, 49, -1)
        relative_position_bias_63 = None
        permute_95 = relative_position_bias_64.permute(2, 0, 1)
        relative_position_bias_64 = None
        contiguous_42 = permute_95.contiguous()
        permute_95 = None
        relative_position_bias_65 = contiguous_42.unsqueeze(0)
        contiguous_42 = None
        x_259 = torch._C._nn.pad(layer_norm_45, (0, 0, 0, 0, 0, 0), "constant", None)
        layer_norm_45 = None
        x_260 = torch.roll(x_259, shifts=(-3, -3), dims=(1, 2))
        x_259 = None
        x_261 = x_260.view(1, 2, 7, 2, 7, 512)
        x_260 = None
        permute_96 = x_261.permute(0, 1, 3, 2, 4, 5)
        x_261 = None
        x_262 = permute_96.reshape(4, 49, 512)
        permute_96 = None
        qkv_42 = torch._C._nn.linear(
            x_262,
            l_self_modules_features_modules_5_modules_17_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_features_modules_5_modules_17_modules_attn_modules_qkv_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_17_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_features_modules_5_modules_17_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_95 = qkv_42.reshape(4, 49, 3, 16, 32)
        qkv_42 = None
        qkv_43 = reshape_95.permute(2, 0, 3, 1, 4)
        reshape_95 = None
        q_42 = qkv_43[0]
        k_21 = qkv_43[1]
        v_21 = qkv_43[2]
        qkv_43 = None
        q_43 = q_42 * 0.1767766952966369
        q_42 = None
        transpose_42 = k_21.transpose(-2, -1)
        k_21 = None
        attn_114 = q_43.matmul(transpose_42)
        q_43 = transpose_42 = None
        attn_115 = attn_114 + relative_position_bias_65
        attn_114 = relative_position_bias_65 = None
        attn_mask_50 = x_262.new_zeros((14, 14))
        x_262 = None
        attn_mask_50[(slice(0, -7, None), slice(0, -7, None))] = 0
        setitem_90 = attn_mask_50
        setitem_90 = None
        attn_mask_50[(slice(0, -7, None), slice(-7, -3, None))] = 1
        setitem_91 = attn_mask_50
        setitem_91 = None
        attn_mask_50[(slice(0, -7, None), slice(-3, None, None))] = 2
        setitem_92 = attn_mask_50
        setitem_92 = None
        attn_mask_50[(slice(-7, -3, None), slice(0, -7, None))] = 3
        setitem_93 = attn_mask_50
        setitem_93 = None
        attn_mask_50[(slice(-7, -3, None), slice(-7, -3, None))] = 4
        setitem_94 = attn_mask_50
        setitem_94 = None
        attn_mask_50[(slice(-7, -3, None), slice(-3, None, None))] = 5
        setitem_95 = attn_mask_50
        setitem_95 = None
        attn_mask_50[(slice(-3, None, None), slice(0, -7, None))] = 6
        setitem_96 = attn_mask_50
        setitem_96 = None
        attn_mask_50[(slice(-3, None, None), slice(-7, -3, None))] = 7
        setitem_97 = attn_mask_50
        setitem_97 = None
        attn_mask_50[(slice(-3, None, None), slice(-3, None, None))] = 8
        setitem_98 = attn_mask_50
        setitem_98 = None
        attn_mask_51 = attn_mask_50.view(2, 7, 2, 7)
        attn_mask_50 = None
        permute_98 = attn_mask_51.permute(0, 2, 1, 3)
        attn_mask_51 = None
        attn_mask_52 = permute_98.reshape(4, 49)
        permute_98 = None
        unsqueeze_62 = attn_mask_52.unsqueeze(1)
        unsqueeze_63 = attn_mask_52.unsqueeze(2)
        attn_mask_52 = None
        attn_mask_53 = unsqueeze_62 - unsqueeze_63
        unsqueeze_62 = unsqueeze_63 = None
        ne_10 = attn_mask_53 != 0
        masked_fill_20 = attn_mask_53.masked_fill(ne_10, -100.0)
        ne_10 = None
        eq_10 = attn_mask_53 == 0
        attn_mask_53 = None
        attn_mask_54 = masked_fill_20.masked_fill(eq_10, 0.0)
        masked_fill_20 = eq_10 = None
        attn_116 = attn_115.view(1, 4, 16, 49, 49)
        attn_115 = None
        unsqueeze_64 = attn_mask_54.unsqueeze(1)
        attn_mask_54 = None
        unsqueeze_65 = unsqueeze_64.unsqueeze(0)
        unsqueeze_64 = None
        attn_117 = attn_116 + unsqueeze_65
        attn_116 = unsqueeze_65 = None
        attn_118 = attn_117.view(-1, 16, 49, 49)
        attn_117 = None
        attn_119 = torch.nn.functional.softmax(attn_118, dim=-1)
        attn_118 = None
        attn_120 = torch.nn.functional.dropout(attn_119, p=0.0, training=False)
        attn_119 = None
        matmul_43 = attn_120.matmul(v_21)
        attn_120 = v_21 = None
        transpose_43 = matmul_43.transpose(1, 2)
        matmul_43 = None
        x_263 = transpose_43.reshape(4, 49, 512)
        transpose_43 = None
        x_264 = torch._C._nn.linear(
            x_263,
            l_self_modules_features_modules_5_modules_17_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_5_modules_17_modules_attn_modules_proj_parameters_bias_,
        )
        x_263 = l_self_modules_features_modules_5_modules_17_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_5_modules_17_modules_attn_modules_proj_parameters_bias_ = (None)
        x_265 = torch.nn.functional.dropout(x_264, p=0.0, training=False)
        x_264 = None
        x_266 = x_265.view(1, 2, 2, 7, 7, 512)
        x_265 = None
        permute_99 = x_266.permute(0, 1, 3, 2, 4, 5)
        x_266 = None
        x_267 = permute_99.reshape(1, 14, 14, 512)
        permute_99 = None
        x_268 = torch.roll(x_267, shifts=(3, 3), dims=(1, 2))
        x_267 = None
        getitem_117 = x_268[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_268 = None
        x_269 = getitem_117.contiguous()
        getitem_117 = None
        _log_api_usage_once_42 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_42 = None
        x_270 = x_258 + x_269
        x_258 = x_269 = None
        layer_norm_46 = torch.nn.functional.layer_norm(
            x_270,
            (512,),
            l_self_modules_features_modules_5_modules_17_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_5_modules_17_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_5_modules_17_modules_norm2_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_17_modules_norm2_parameters_bias_
        ) = None
        input_109 = torch._C._nn.linear(
            layer_norm_46,
            l_self_modules_features_modules_5_modules_17_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_17_modules_mlp_modules_0_parameters_bias_,
        )
        layer_norm_46 = l_self_modules_features_modules_5_modules_17_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_17_modules_mlp_modules_0_parameters_bias_ = (None)
        input_110 = torch._C._nn.gelu(input_109, approximate="none")
        input_109 = None
        input_111 = torch.nn.functional.dropout(input_110, 0.0, False, False)
        input_110 = None
        input_112 = torch._C._nn.linear(
            input_111,
            l_self_modules_features_modules_5_modules_17_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_17_modules_mlp_modules_3_parameters_bias_,
        )
        input_111 = l_self_modules_features_modules_5_modules_17_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_17_modules_mlp_modules_3_parameters_bias_ = (None)
        input_113 = torch.nn.functional.dropout(input_112, 0.0, False, False)
        input_112 = None
        _log_api_usage_once_43 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_43 = None
        x_271 = x_270 + input_113
        x_270 = input_113 = None
        x_272 = torch._C._nn.pad(x_271, (0, 0, 0, 0, 0, 0), "constant", None)
        x_271 = None
        x0_2 = x_272[
            (Ellipsis, slice(0, None, 2), slice(0, None, 2), slice(None, None, None))
        ]
        x1_2 = x_272[
            (Ellipsis, slice(1, None, 2), slice(0, None, 2), slice(None, None, None))
        ]
        x2_2 = x_272[
            (Ellipsis, slice(0, None, 2), slice(1, None, 2), slice(None, None, None))
        ]
        x3_2 = x_272[
            (Ellipsis, slice(1, None, 2), slice(1, None, 2), slice(None, None, None))
        ]
        x_272 = None
        x_273 = torch.cat([x0_2, x1_2, x2_2, x3_2], -1)
        x0_2 = x1_2 = x2_2 = x3_2 = None
        x_274 = torch.nn.functional.layer_norm(
            x_273,
            (2048,),
            l_self_modules_features_modules_6_modules_norm_parameters_weight_,
            l_self_modules_features_modules_6_modules_norm_parameters_bias_,
            1e-05,
        )
        x_273 = (
            l_self_modules_features_modules_6_modules_norm_parameters_weight_
        ) = l_self_modules_features_modules_6_modules_norm_parameters_bias_ = None
        x_275 = torch._C._nn.linear(
            x_274,
            l_self_modules_features_modules_6_modules_reduction_parameters_weight_,
            None,
        )
        x_274 = (
            l_self_modules_features_modules_6_modules_reduction_parameters_weight_
        ) = None
        layer_norm_48 = torch.nn.functional.layer_norm(
            x_275,
            (1024,),
            l_self_modules_features_modules_7_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_7_modules_0_modules_norm1_parameters_weight_ = (
            l_self_modules_features_modules_7_modules_0_modules_norm1_parameters_bias_
        ) = None
        relative_position_bias_66 = l_self_modules_features_modules_7_modules_0_modules_attn_parameters_relative_position_bias_table_[
            l_self_modules_features_modules_7_modules_0_modules_attn_buffers_relative_position_index_
        ]
        l_self_modules_features_modules_7_modules_0_modules_attn_parameters_relative_position_bias_table_ = l_self_modules_features_modules_7_modules_0_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_67 = relative_position_bias_66.view(49, 49, -1)
        relative_position_bias_66 = None
        permute_100 = relative_position_bias_67.permute(2, 0, 1)
        relative_position_bias_67 = None
        contiguous_44 = permute_100.contiguous()
        permute_100 = None
        relative_position_bias_68 = contiguous_44.unsqueeze(0)
        contiguous_44 = None
        x_276 = torch._C._nn.pad(layer_norm_48, (0, 0, 0, 0, 0, 0), "constant", None)
        layer_norm_48 = None
        x_277 = x_276.view(1, 1, 7, 1, 7, 1024)
        x_276 = None
        permute_101 = x_277.permute(0, 1, 3, 2, 4, 5)
        x_277 = None
        x_278 = permute_101.reshape(1, 49, 1024)
        permute_101 = None
        qkv_44 = torch._C._nn.linear(
            x_278,
            l_self_modules_features_modules_7_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        x_278 = l_self_modules_features_modules_7_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_100 = qkv_44.reshape(1, 49, 3, 32, 32)
        qkv_44 = None
        qkv_45 = reshape_100.permute(2, 0, 3, 1, 4)
        reshape_100 = None
        q_44 = qkv_45[0]
        k_22 = qkv_45[1]
        v_22 = qkv_45[2]
        qkv_45 = None
        q_45 = q_44 * 0.1767766952966369
        q_44 = None
        transpose_44 = k_22.transpose(-2, -1)
        k_22 = None
        attn_121 = q_45.matmul(transpose_44)
        q_45 = transpose_44 = None
        attn_122 = attn_121 + relative_position_bias_68
        attn_121 = relative_position_bias_68 = None
        attn_123 = torch.nn.functional.softmax(attn_122, dim=-1)
        attn_122 = None
        attn_124 = torch.nn.functional.dropout(attn_123, p=0.0, training=False)
        attn_123 = None
        matmul_45 = attn_124.matmul(v_22)
        attn_124 = v_22 = None
        transpose_45 = matmul_45.transpose(1, 2)
        matmul_45 = None
        x_279 = transpose_45.reshape(1, 49, 1024)
        transpose_45 = None
        x_280 = torch._C._nn.linear(
            x_279,
            l_self_modules_features_modules_7_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_279 = l_self_modules_features_modules_7_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_281 = torch.nn.functional.dropout(x_280, p=0.0, training=False)
        x_280 = None
        x_282 = x_281.view(1, 1, 1, 7, 7, 1024)
        x_281 = None
        permute_103 = x_282.permute(0, 1, 3, 2, 4, 5)
        x_282 = None
        x_283 = permute_103.reshape(1, 7, 7, 1024)
        permute_103 = None
        getitem_126 = x_283[
            (
                slice(None, None, None),
                slice(None, 7, None),
                slice(None, 7, None),
                slice(None, None, None),
            )
        ]
        x_283 = None
        x_284 = getitem_126.contiguous()
        getitem_126 = None
        _log_api_usage_once_44 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_44 = None
        x_285 = x_275 + x_284
        x_275 = x_284 = None
        layer_norm_49 = torch.nn.functional.layer_norm(
            x_285,
            (1024,),
            l_self_modules_features_modules_7_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_7_modules_0_modules_norm2_parameters_weight_ = (
            l_self_modules_features_modules_7_modules_0_modules_norm2_parameters_bias_
        ) = None
        input_114 = torch._C._nn.linear(
            layer_norm_49,
            l_self_modules_features_modules_7_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        layer_norm_49 = l_self_modules_features_modules_7_modules_0_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_mlp_modules_0_parameters_bias_ = (None)
        input_115 = torch._C._nn.gelu(input_114, approximate="none")
        input_114 = None
        input_116 = torch.nn.functional.dropout(input_115, 0.0, False, False)
        input_115 = None
        input_117 = torch._C._nn.linear(
            input_116,
            l_self_modules_features_modules_7_modules_0_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_mlp_modules_3_parameters_bias_,
        )
        input_116 = l_self_modules_features_modules_7_modules_0_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_mlp_modules_3_parameters_bias_ = (None)
        input_118 = torch.nn.functional.dropout(input_117, 0.0, False, False)
        input_117 = None
        _log_api_usage_once_45 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_45 = None
        x_286 = x_285 + input_118
        x_285 = input_118 = None
        layer_norm_50 = torch.nn.functional.layer_norm(
            x_286,
            (1024,),
            l_self_modules_features_modules_7_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_7_modules_1_modules_norm1_parameters_weight_ = (
            l_self_modules_features_modules_7_modules_1_modules_norm1_parameters_bias_
        ) = None
        relative_position_bias_69 = l_self_modules_features_modules_7_modules_1_modules_attn_parameters_relative_position_bias_table_[
            l_self_modules_features_modules_7_modules_1_modules_attn_buffers_relative_position_index_
        ]
        l_self_modules_features_modules_7_modules_1_modules_attn_parameters_relative_position_bias_table_ = l_self_modules_features_modules_7_modules_1_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_70 = relative_position_bias_69.view(49, 49, -1)
        relative_position_bias_69 = None
        permute_104 = relative_position_bias_70.permute(2, 0, 1)
        relative_position_bias_70 = None
        contiguous_46 = permute_104.contiguous()
        permute_104 = None
        relative_position_bias_71 = contiguous_46.unsqueeze(0)
        contiguous_46 = None
        x_287 = torch._C._nn.pad(layer_norm_50, (0, 0, 0, 0, 0, 0), "constant", None)
        layer_norm_50 = None
        x_288 = x_287.view(1, 1, 7, 1, 7, 1024)
        x_287 = None
        permute_105 = x_288.permute(0, 1, 3, 2, 4, 5)
        x_288 = None
        x_289 = permute_105.reshape(1, 49, 1024)
        permute_105 = None
        qkv_46 = torch._C._nn.linear(
            x_289,
            l_self_modules_features_modules_7_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_attn_modules_qkv_parameters_bias_,
        )
        x_289 = l_self_modules_features_modules_7_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_104 = qkv_46.reshape(1, 49, 3, 32, 32)
        qkv_46 = None
        qkv_47 = reshape_104.permute(2, 0, 3, 1, 4)
        reshape_104 = None
        q_46 = qkv_47[0]
        k_23 = qkv_47[1]
        v_23 = qkv_47[2]
        qkv_47 = None
        q_47 = q_46 * 0.1767766952966369
        q_46 = None
        transpose_46 = k_23.transpose(-2, -1)
        k_23 = None
        attn_125 = q_47.matmul(transpose_46)
        q_47 = transpose_46 = None
        attn_126 = attn_125 + relative_position_bias_71
        attn_125 = relative_position_bias_71 = None
        attn_127 = torch.nn.functional.softmax(attn_126, dim=-1)
        attn_126 = None
        attn_128 = torch.nn.functional.dropout(attn_127, p=0.0, training=False)
        attn_127 = None
        matmul_47 = attn_128.matmul(v_23)
        attn_128 = v_23 = None
        transpose_47 = matmul_47.transpose(1, 2)
        matmul_47 = None
        x_290 = transpose_47.reshape(1, 49, 1024)
        transpose_47 = None
        x_291 = torch._C._nn.linear(
            x_290,
            l_self_modules_features_modules_7_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_290 = l_self_modules_features_modules_7_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_292 = torch.nn.functional.dropout(x_291, p=0.0, training=False)
        x_291 = None
        x_293 = x_292.view(1, 1, 1, 7, 7, 1024)
        x_292 = None
        permute_107 = x_293.permute(0, 1, 3, 2, 4, 5)
        x_293 = None
        x_294 = permute_107.reshape(1, 7, 7, 1024)
        permute_107 = None
        getitem_131 = x_294[
            (
                slice(None, None, None),
                slice(None, 7, None),
                slice(None, 7, None),
                slice(None, None, None),
            )
        ]
        x_294 = None
        x_295 = getitem_131.contiguous()
        getitem_131 = None
        _log_api_usage_once_46 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_46 = None
        x_296 = x_286 + x_295
        x_286 = x_295 = None
        layer_norm_51 = torch.nn.functional.layer_norm(
            x_296,
            (1024,),
            l_self_modules_features_modules_7_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_features_modules_7_modules_1_modules_norm2_parameters_weight_ = (
            l_self_modules_features_modules_7_modules_1_modules_norm2_parameters_bias_
        ) = None
        input_119 = torch._C._nn.linear(
            layer_norm_51,
            l_self_modules_features_modules_7_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        layer_norm_51 = l_self_modules_features_modules_7_modules_1_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_mlp_modules_0_parameters_bias_ = (None)
        input_120 = torch._C._nn.gelu(input_119, approximate="none")
        input_119 = None
        input_121 = torch.nn.functional.dropout(input_120, 0.0, False, False)
        input_120 = None
        input_122 = torch._C._nn.linear(
            input_121,
            l_self_modules_features_modules_7_modules_1_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_mlp_modules_3_parameters_bias_,
        )
        input_121 = l_self_modules_features_modules_7_modules_1_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_mlp_modules_3_parameters_bias_ = (None)
        input_123 = torch.nn.functional.dropout(input_122, 0.0, False, False)
        input_122 = None
        _log_api_usage_once_47 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_47 = None
        x_297 = x_296 + input_123
        x_296 = input_123 = None
        x_298 = torch.nn.functional.layer_norm(
            x_297,
            (1024,),
            l_self_modules_norm_parameters_weight_,
            l_self_modules_norm_parameters_bias_,
            1e-05,
        )
        x_297 = (
            l_self_modules_norm_parameters_weight_
        ) = l_self_modules_norm_parameters_bias_ = None
        x_299 = torch.permute(x_298, [0, 3, 1, 2])
        x_298 = None
        x_300 = torch.nn.functional.adaptive_avg_pool2d(x_299, 1)
        x_299 = None
        x_301 = x_300.flatten(1, -1)
        x_300 = None
        x_302 = torch._C._nn.linear(
            x_301,
            l_self_modules_head_parameters_weight_,
            l_self_modules_head_parameters_bias_,
        )
        x_301 = (
            l_self_modules_head_parameters_weight_
        ) = l_self_modules_head_parameters_bias_ = None
        return (x_302,)
