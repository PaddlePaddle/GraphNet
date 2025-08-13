import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_features_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_x_: torch.Tensor,
        L_self_modules_features_modules_0_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_0_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_attn_buffers_relative_coords_table_: torch.Tensor,
        L_self_modules_features_modules_1_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_attn_buffers_relative_coords_table_: torch.Tensor,
        L_self_modules_features_modules_1_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_reduction_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_attn_buffers_relative_coords_table_: torch.Tensor,
        L_self_modules_features_modules_3_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_3_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_attn_buffers_relative_coords_table_: torch.Tensor,
        L_self_modules_features_modules_3_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_3_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_reduction_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_attn_buffers_relative_coords_table_: torch.Tensor,
        L_self_modules_features_modules_5_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_5_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_attn_buffers_relative_coords_table_: torch.Tensor,
        L_self_modules_features_modules_5_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_5_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_attn_buffers_relative_coords_table_: torch.Tensor,
        L_self_modules_features_modules_5_modules_2_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_5_modules_2_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_attn_buffers_relative_coords_table_: torch.Tensor,
        L_self_modules_features_modules_5_modules_3_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_5_modules_3_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_attn_buffers_relative_coords_table_: torch.Tensor,
        L_self_modules_features_modules_5_modules_4_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_5_modules_4_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_attn_buffers_relative_coords_table_: torch.Tensor,
        L_self_modules_features_modules_5_modules_5_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_5_modules_5_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_reduction_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_attn_buffers_relative_coords_table_: torch.Tensor,
        L_self_modules_features_modules_7_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_7_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_attn_buffers_relative_coords_table_: torch.Tensor,
        L_self_modules_features_modules_7_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_7_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_features_modules_1_modules_0_modules_attn_buffers_relative_coords_table_ = L_self_modules_features_modules_1_modules_0_modules_attn_buffers_relative_coords_table_
        l_self_modules_features_modules_1_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_1_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_1_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_1_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_1_modules_0_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = L_self_modules_features_modules_1_modules_0_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_
        l_self_modules_features_modules_1_modules_0_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_1_modules_0_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_1_modules_0_modules_attn_parameters_logit_scale_ = L_self_modules_features_modules_1_modules_0_modules_attn_parameters_logit_scale_
        l_self_modules_features_modules_1_modules_0_modules_norm1_parameters_weight_ = (
            L_self_modules_features_modules_1_modules_0_modules_norm1_parameters_weight_
        )
        l_self_modules_features_modules_1_modules_0_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_1_modules_0_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_1_modules_0_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_1_modules_0_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_1_modules_0_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_1_modules_0_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_1_modules_0_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_1_modules_0_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_1_modules_0_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_1_modules_0_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_1_modules_0_modules_norm2_parameters_weight_ = (
            L_self_modules_features_modules_1_modules_0_modules_norm2_parameters_weight_
        )
        l_self_modules_features_modules_1_modules_0_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_1_modules_0_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_1_modules_1_modules_attn_buffers_relative_coords_table_ = L_self_modules_features_modules_1_modules_1_modules_attn_buffers_relative_coords_table_
        l_self_modules_features_modules_1_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_1_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_1_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_1_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_1_modules_1_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = L_self_modules_features_modules_1_modules_1_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_
        l_self_modules_features_modules_1_modules_1_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_1_modules_1_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_1_modules_1_modules_attn_parameters_logit_scale_ = L_self_modules_features_modules_1_modules_1_modules_attn_parameters_logit_scale_
        l_self_modules_features_modules_1_modules_1_modules_norm1_parameters_weight_ = (
            L_self_modules_features_modules_1_modules_1_modules_norm1_parameters_weight_
        )
        l_self_modules_features_modules_1_modules_1_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_1_modules_1_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_1_modules_1_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_1_modules_1_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_1_modules_1_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_1_modules_1_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_1_modules_1_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_1_modules_1_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_1_modules_1_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_1_modules_1_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_1_modules_1_modules_norm2_parameters_weight_ = (
            L_self_modules_features_modules_1_modules_1_modules_norm2_parameters_weight_
        )
        l_self_modules_features_modules_1_modules_1_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_1_modules_1_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_2_modules_reduction_parameters_weight_ = (
            L_self_modules_features_modules_2_modules_reduction_parameters_weight_
        )
        l_self_modules_features_modules_2_modules_norm_parameters_weight_ = (
            L_self_modules_features_modules_2_modules_norm_parameters_weight_
        )
        l_self_modules_features_modules_2_modules_norm_parameters_bias_ = (
            L_self_modules_features_modules_2_modules_norm_parameters_bias_
        )
        l_self_modules_features_modules_3_modules_0_modules_attn_buffers_relative_coords_table_ = L_self_modules_features_modules_3_modules_0_modules_attn_buffers_relative_coords_table_
        l_self_modules_features_modules_3_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_3_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_3_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_3_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_3_modules_0_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = L_self_modules_features_modules_3_modules_0_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_
        l_self_modules_features_modules_3_modules_0_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_3_modules_0_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_3_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_3_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_3_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_3_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_3_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_3_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_3_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_3_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_3_modules_0_modules_attn_parameters_logit_scale_ = L_self_modules_features_modules_3_modules_0_modules_attn_parameters_logit_scale_
        l_self_modules_features_modules_3_modules_0_modules_norm1_parameters_weight_ = (
            L_self_modules_features_modules_3_modules_0_modules_norm1_parameters_weight_
        )
        l_self_modules_features_modules_3_modules_0_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_3_modules_0_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_3_modules_0_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_3_modules_0_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_3_modules_0_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_3_modules_0_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_3_modules_0_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_3_modules_0_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_3_modules_0_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_3_modules_0_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_3_modules_0_modules_norm2_parameters_weight_ = (
            L_self_modules_features_modules_3_modules_0_modules_norm2_parameters_weight_
        )
        l_self_modules_features_modules_3_modules_0_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_3_modules_0_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_3_modules_1_modules_attn_buffers_relative_coords_table_ = L_self_modules_features_modules_3_modules_1_modules_attn_buffers_relative_coords_table_
        l_self_modules_features_modules_3_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_3_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_3_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_3_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_3_modules_1_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = L_self_modules_features_modules_3_modules_1_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_
        l_self_modules_features_modules_3_modules_1_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_3_modules_1_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_3_modules_1_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_3_modules_1_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_3_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_3_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_3_modules_1_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_3_modules_1_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_3_modules_1_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_3_modules_1_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_3_modules_1_modules_attn_parameters_logit_scale_ = L_self_modules_features_modules_3_modules_1_modules_attn_parameters_logit_scale_
        l_self_modules_features_modules_3_modules_1_modules_norm1_parameters_weight_ = (
            L_self_modules_features_modules_3_modules_1_modules_norm1_parameters_weight_
        )
        l_self_modules_features_modules_3_modules_1_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_3_modules_1_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_3_modules_1_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_3_modules_1_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_3_modules_1_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_3_modules_1_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_3_modules_1_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_3_modules_1_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_3_modules_1_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_3_modules_1_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_3_modules_1_modules_norm2_parameters_weight_ = (
            L_self_modules_features_modules_3_modules_1_modules_norm2_parameters_weight_
        )
        l_self_modules_features_modules_3_modules_1_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_3_modules_1_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_4_modules_reduction_parameters_weight_ = (
            L_self_modules_features_modules_4_modules_reduction_parameters_weight_
        )
        l_self_modules_features_modules_4_modules_norm_parameters_weight_ = (
            L_self_modules_features_modules_4_modules_norm_parameters_weight_
        )
        l_self_modules_features_modules_4_modules_norm_parameters_bias_ = (
            L_self_modules_features_modules_4_modules_norm_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_0_modules_attn_buffers_relative_coords_table_ = L_self_modules_features_modules_5_modules_0_modules_attn_buffers_relative_coords_table_
        l_self_modules_features_modules_5_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_0_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_0_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_0_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_5_modules_0_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_5_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_5_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_5_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_5_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_5_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_5_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_5_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_5_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_5_modules_0_modules_attn_parameters_logit_scale_ = L_self_modules_features_modules_5_modules_0_modules_attn_parameters_logit_scale_
        l_self_modules_features_modules_5_modules_0_modules_norm1_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_0_modules_norm1_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_0_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_0_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_0_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_0_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_0_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_0_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_0_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_0_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_0_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_0_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_0_modules_norm2_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_0_modules_norm2_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_0_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_0_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_1_modules_attn_buffers_relative_coords_table_ = L_self_modules_features_modules_5_modules_1_modules_attn_buffers_relative_coords_table_
        l_self_modules_features_modules_5_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_1_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_1_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_1_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_5_modules_1_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_5_modules_1_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_5_modules_1_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_5_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_5_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_5_modules_1_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_5_modules_1_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_5_modules_1_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_5_modules_1_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_5_modules_1_modules_attn_parameters_logit_scale_ = L_self_modules_features_modules_5_modules_1_modules_attn_parameters_logit_scale_
        l_self_modules_features_modules_5_modules_1_modules_norm1_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_1_modules_norm1_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_1_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_1_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_1_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_1_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_1_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_1_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_1_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_1_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_1_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_1_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_1_modules_norm2_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_1_modules_norm2_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_1_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_1_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_2_modules_attn_buffers_relative_coords_table_ = L_self_modules_features_modules_5_modules_2_modules_attn_buffers_relative_coords_table_
        l_self_modules_features_modules_5_modules_2_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_2_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_2_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_2_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_2_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_2_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_2_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_5_modules_2_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_5_modules_2_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_5_modules_2_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_5_modules_2_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_5_modules_2_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_5_modules_2_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_5_modules_2_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_5_modules_2_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_5_modules_2_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_5_modules_2_modules_attn_parameters_logit_scale_ = L_self_modules_features_modules_5_modules_2_modules_attn_parameters_logit_scale_
        l_self_modules_features_modules_5_modules_2_modules_norm1_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_2_modules_norm1_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_2_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_2_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_2_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_2_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_2_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_2_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_2_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_2_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_2_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_2_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_2_modules_norm2_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_2_modules_norm2_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_2_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_2_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_3_modules_attn_buffers_relative_coords_table_ = L_self_modules_features_modules_5_modules_3_modules_attn_buffers_relative_coords_table_
        l_self_modules_features_modules_5_modules_3_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_3_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_3_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_3_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_3_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_3_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_3_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_5_modules_3_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_5_modules_3_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_5_modules_3_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_5_modules_3_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_5_modules_3_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_5_modules_3_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_5_modules_3_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_5_modules_3_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_5_modules_3_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_5_modules_3_modules_attn_parameters_logit_scale_ = L_self_modules_features_modules_5_modules_3_modules_attn_parameters_logit_scale_
        l_self_modules_features_modules_5_modules_3_modules_norm1_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_3_modules_norm1_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_3_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_3_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_3_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_3_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_3_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_3_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_3_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_3_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_3_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_3_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_3_modules_norm2_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_3_modules_norm2_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_3_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_3_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_4_modules_attn_buffers_relative_coords_table_ = L_self_modules_features_modules_5_modules_4_modules_attn_buffers_relative_coords_table_
        l_self_modules_features_modules_5_modules_4_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_4_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_4_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_4_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_4_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_4_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_4_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_5_modules_4_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_5_modules_4_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_5_modules_4_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_5_modules_4_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_5_modules_4_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_5_modules_4_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_5_modules_4_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_5_modules_4_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_5_modules_4_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_5_modules_4_modules_attn_parameters_logit_scale_ = L_self_modules_features_modules_5_modules_4_modules_attn_parameters_logit_scale_
        l_self_modules_features_modules_5_modules_4_modules_norm1_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_4_modules_norm1_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_4_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_4_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_4_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_4_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_4_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_4_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_4_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_4_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_4_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_4_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_4_modules_norm2_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_4_modules_norm2_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_4_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_4_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_5_modules_attn_buffers_relative_coords_table_ = L_self_modules_features_modules_5_modules_5_modules_attn_buffers_relative_coords_table_
        l_self_modules_features_modules_5_modules_5_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_5_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_5_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_5_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_5_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_5_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_5_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_5_modules_5_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_5_modules_5_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_5_modules_5_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_5_modules_5_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_5_modules_5_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_5_modules_5_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_5_modules_5_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_5_modules_5_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_5_modules_5_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_5_modules_5_modules_attn_parameters_logit_scale_ = L_self_modules_features_modules_5_modules_5_modules_attn_parameters_logit_scale_
        l_self_modules_features_modules_5_modules_5_modules_norm1_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_5_modules_norm1_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_5_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_5_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_5_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_5_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_5_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_5_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_5_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_5_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_5_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_5_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_5_modules_norm2_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_5_modules_norm2_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_5_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_5_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_6_modules_reduction_parameters_weight_ = (
            L_self_modules_features_modules_6_modules_reduction_parameters_weight_
        )
        l_self_modules_features_modules_6_modules_norm_parameters_weight_ = (
            L_self_modules_features_modules_6_modules_norm_parameters_weight_
        )
        l_self_modules_features_modules_6_modules_norm_parameters_bias_ = (
            L_self_modules_features_modules_6_modules_norm_parameters_bias_
        )
        l_self_modules_features_modules_7_modules_0_modules_attn_buffers_relative_coords_table_ = L_self_modules_features_modules_7_modules_0_modules_attn_buffers_relative_coords_table_
        l_self_modules_features_modules_7_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_7_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_7_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_7_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_7_modules_0_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = L_self_modules_features_modules_7_modules_0_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_
        l_self_modules_features_modules_7_modules_0_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_7_modules_0_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_7_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_7_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_7_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_7_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_7_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_7_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_7_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_7_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_7_modules_0_modules_attn_parameters_logit_scale_ = L_self_modules_features_modules_7_modules_0_modules_attn_parameters_logit_scale_
        l_self_modules_features_modules_7_modules_0_modules_norm1_parameters_weight_ = (
            L_self_modules_features_modules_7_modules_0_modules_norm1_parameters_weight_
        )
        l_self_modules_features_modules_7_modules_0_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_7_modules_0_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_7_modules_0_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_7_modules_0_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_7_modules_0_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_7_modules_0_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_7_modules_0_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_7_modules_0_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_7_modules_0_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_7_modules_0_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_7_modules_0_modules_norm2_parameters_weight_ = (
            L_self_modules_features_modules_7_modules_0_modules_norm2_parameters_weight_
        )
        l_self_modules_features_modules_7_modules_0_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_7_modules_0_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_7_modules_1_modules_attn_buffers_relative_coords_table_ = L_self_modules_features_modules_7_modules_1_modules_attn_buffers_relative_coords_table_
        l_self_modules_features_modules_7_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_7_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_7_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_7_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_7_modules_1_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = L_self_modules_features_modules_7_modules_1_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_
        l_self_modules_features_modules_7_modules_1_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_7_modules_1_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_7_modules_1_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_7_modules_1_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_7_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_7_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_7_modules_1_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_7_modules_1_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_7_modules_1_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_7_modules_1_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_7_modules_1_modules_attn_parameters_logit_scale_ = L_self_modules_features_modules_7_modules_1_modules_attn_parameters_logit_scale_
        l_self_modules_features_modules_7_modules_1_modules_norm1_parameters_weight_ = (
            L_self_modules_features_modules_7_modules_1_modules_norm1_parameters_weight_
        )
        l_self_modules_features_modules_7_modules_1_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_7_modules_1_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_7_modules_1_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_7_modules_1_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_7_modules_1_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_7_modules_1_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_7_modules_1_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_7_modules_1_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_7_modules_1_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_7_modules_1_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_7_modules_1_modules_norm2_parameters_weight_ = (
            L_self_modules_features_modules_7_modules_1_modules_norm2_parameters_weight_
        )
        l_self_modules_features_modules_7_modules_1_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_7_modules_1_modules_norm2_parameters_bias_
        )
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
            (96,),
            l_self_modules_features_modules_0_modules_2_parameters_weight_,
            l_self_modules_features_modules_0_modules_2_parameters_bias_,
            1e-05,
        )
        input_2 = (
            l_self_modules_features_modules_0_modules_2_parameters_weight_
        ) = l_self_modules_features_modules_0_modules_2_parameters_bias_ = None
        input_4 = torch._C._nn.linear(
            l_self_modules_features_modules_1_modules_0_modules_attn_buffers_relative_coords_table_,
            l_self_modules_features_modules_1_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_1_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_1_modules_0_modules_attn_buffers_relative_coords_table_ = l_self_modules_features_modules_1_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_1_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = (None)
        input_5 = torch.nn.functional.relu(input_4, inplace=True)
        input_4 = None
        input_6 = torch._C._nn.linear(
            input_5,
            l_self_modules_features_modules_1_modules_0_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_,
            None,
        )
        input_5 = l_self_modules_features_modules_1_modules_0_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = (None)
        view = input_6.view(-1, 3)
        input_6 = None
        relative_position_bias = view[
            l_self_modules_features_modules_1_modules_0_modules_attn_buffers_relative_position_index_
        ]
        view = l_self_modules_features_modules_1_modules_0_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_1 = relative_position_bias.view(64, 64, -1)
        relative_position_bias = None
        permute_1 = relative_position_bias_1.permute(2, 0, 1)
        relative_position_bias_1 = None
        contiguous = permute_1.contiguous()
        permute_1 = None
        relative_position_bias_2 = contiguous.unsqueeze(0)
        contiguous = None
        sigmoid = torch.sigmoid(relative_position_bias_2)
        relative_position_bias_2 = None
        relative_position_bias_3 = 16 * sigmoid
        sigmoid = None
        x = torch._C._nn.pad(input_3, (0, 0, 0, 0, 0, 0), "constant", None)
        x_1 = x.view(1, 7, 8, 7, 8, 96)
        x = None
        permute_2 = x_1.permute(0, 1, 3, 2, 4, 5)
        x_1 = None
        x_2 = permute_2.reshape(49, 64, 96)
        permute_2 = None
        qkv_bias = (
            l_self_modules_features_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_.clone()
        )
        l_self_modules_features_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_ = (
            None
        )
        getitem_1 = qkv_bias[slice(96, 192, None)]
        zero_ = getitem_1.zero_()
        getitem_1 = zero_ = None
        qkv = torch._C._nn.linear(
            x_2,
            l_self_modules_features_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_,
            qkv_bias,
        )
        x_2 = l_self_modules_features_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_ = (qkv_bias) = (
            None
        )
        reshape_1 = qkv.reshape(49, 64, 3, 3, 32)
        qkv = None
        qkv_1 = reshape_1.permute(2, 0, 3, 1, 4)
        reshape_1 = None
        q = qkv_1[0]
        k = qkv_1[1]
        v = qkv_1[2]
        qkv_1 = None
        normalize = torch.nn.functional.normalize(q, dim=-1)
        q = None
        normalize_1 = torch.nn.functional.normalize(k, dim=-1)
        k = None
        transpose = normalize_1.transpose(-2, -1)
        normalize_1 = None
        attn = normalize @ transpose
        normalize = transpose = None
        clamp = torch.clamp(
            l_self_modules_features_modules_1_modules_0_modules_attn_parameters_logit_scale_,
            max=4.605170185988092,
        )
        l_self_modules_features_modules_1_modules_0_modules_attn_parameters_logit_scale_ = (
            None
        )
        logit_scale = clamp.exp()
        clamp = None
        attn_1 = attn * logit_scale
        attn = logit_scale = None
        attn_2 = attn_1 + relative_position_bias_3
        attn_1 = relative_position_bias_3 = None
        attn_3 = torch.nn.functional.softmax(attn_2, dim=-1)
        attn_2 = None
        attn_4 = torch.nn.functional.dropout(attn_3, p=0.0, training=False)
        attn_3 = None
        matmul_1 = attn_4.matmul(v)
        attn_4 = v = None
        transpose_1 = matmul_1.transpose(1, 2)
        matmul_1 = None
        x_3 = transpose_1.reshape(49, 64, 96)
        transpose_1 = None
        x_4 = torch._C._nn.linear(
            x_3,
            l_self_modules_features_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_3 = l_self_modules_features_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_5 = torch.nn.functional.dropout(x_4, p=0.0, training=False)
        x_4 = None
        x_6 = x_5.view(1, 7, 7, 8, 8, 96)
        x_5 = None
        permute_4 = x_6.permute(0, 1, 3, 2, 4, 5)
        x_6 = None
        x_7 = permute_4.reshape(1, 56, 56, 96)
        permute_4 = None
        getitem_5 = x_7[
            (
                slice(None, None, None),
                slice(None, 56, None),
                slice(None, 56, None),
                slice(None, None, None),
            )
        ]
        x_7 = None
        x_8 = getitem_5.contiguous()
        getitem_5 = None
        layer_norm_1 = torch.nn.functional.layer_norm(
            x_8,
            (96,),
            l_self_modules_features_modules_1_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_1_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_8 = (
            l_self_modules_features_modules_1_modules_0_modules_norm1_parameters_weight_
        ) = (
            l_self_modules_features_modules_1_modules_0_modules_norm1_parameters_bias_
        ) = None
        _log_api_usage_once = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once = None
        x_9 = input_3 + layer_norm_1
        input_3 = layer_norm_1 = None
        input_7 = torch._C._nn.linear(
            x_9,
            l_self_modules_features_modules_1_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_1_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_1_modules_0_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_1_modules_0_modules_mlp_modules_0_parameters_bias_ = (None)
        input_8 = torch._C._nn.gelu(input_7, approximate="none")
        input_7 = None
        input_9 = torch.nn.functional.dropout(input_8, 0.0, False, False)
        input_8 = None
        input_10 = torch._C._nn.linear(
            input_9,
            l_self_modules_features_modules_1_modules_0_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_1_modules_0_modules_mlp_modules_3_parameters_bias_,
        )
        input_9 = l_self_modules_features_modules_1_modules_0_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_1_modules_0_modules_mlp_modules_3_parameters_bias_ = (None)
        input_11 = torch.nn.functional.dropout(input_10, 0.0, False, False)
        input_10 = None
        layer_norm_2 = torch.nn.functional.layer_norm(
            input_11,
            (96,),
            l_self_modules_features_modules_1_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_1_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        input_11 = (
            l_self_modules_features_modules_1_modules_0_modules_norm2_parameters_weight_
        ) = (
            l_self_modules_features_modules_1_modules_0_modules_norm2_parameters_bias_
        ) = None
        _log_api_usage_once_1 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_1 = None
        x_10 = x_9 + layer_norm_2
        x_9 = layer_norm_2 = None
        input_12 = torch._C._nn.linear(
            l_self_modules_features_modules_1_modules_1_modules_attn_buffers_relative_coords_table_,
            l_self_modules_features_modules_1_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_1_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_1_modules_1_modules_attn_buffers_relative_coords_table_ = l_self_modules_features_modules_1_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_1_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = (None)
        input_13 = torch.nn.functional.relu(input_12, inplace=True)
        input_12 = None
        input_14 = torch._C._nn.linear(
            input_13,
            l_self_modules_features_modules_1_modules_1_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_,
            None,
        )
        input_13 = l_self_modules_features_modules_1_modules_1_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = (None)
        view_4 = input_14.view(-1, 3)
        input_14 = None
        relative_position_bias_4 = view_4[
            l_self_modules_features_modules_1_modules_1_modules_attn_buffers_relative_position_index_
        ]
        view_4 = l_self_modules_features_modules_1_modules_1_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_5 = relative_position_bias_4.view(64, 64, -1)
        relative_position_bias_4 = None
        permute_5 = relative_position_bias_5.permute(2, 0, 1)
        relative_position_bias_5 = None
        contiguous_2 = permute_5.contiguous()
        permute_5 = None
        relative_position_bias_6 = contiguous_2.unsqueeze(0)
        contiguous_2 = None
        sigmoid_1 = torch.sigmoid(relative_position_bias_6)
        relative_position_bias_6 = None
        relative_position_bias_7 = 16 * sigmoid_1
        sigmoid_1 = None
        x_11 = torch._C._nn.pad(x_10, (0, 0, 0, 0, 0, 0), "constant", None)
        x_12 = torch.roll(x_11, shifts=(-4, -4), dims=(1, 2))
        x_11 = None
        x_13 = x_12.view(1, 7, 8, 7, 8, 96)
        x_12 = None
        permute_6 = x_13.permute(0, 1, 3, 2, 4, 5)
        x_13 = None
        x_14 = permute_6.reshape(49, 64, 96)
        permute_6 = None
        qkv_bias_1 = (
            l_self_modules_features_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_.clone()
        )
        l_self_modules_features_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_ = (
            None
        )
        getitem_7 = qkv_bias_1[slice(96, 192, None)]
        zero__1 = getitem_7.zero_()
        getitem_7 = zero__1 = None
        qkv_2 = torch._C._nn.linear(
            x_14,
            l_self_modules_features_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_,
            qkv_bias_1,
        )
        l_self_modules_features_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_1
        ) = None
        reshape_5 = qkv_2.reshape(49, 64, 3, 3, 32)
        qkv_2 = None
        qkv_3 = reshape_5.permute(2, 0, 3, 1, 4)
        reshape_5 = None
        q_1 = qkv_3[0]
        k_1 = qkv_3[1]
        v_1 = qkv_3[2]
        qkv_3 = None
        normalize_2 = torch.nn.functional.normalize(q_1, dim=-1)
        q_1 = None
        normalize_3 = torch.nn.functional.normalize(k_1, dim=-1)
        k_1 = None
        transpose_2 = normalize_3.transpose(-2, -1)
        normalize_3 = None
        attn_5 = normalize_2 @ transpose_2
        normalize_2 = transpose_2 = None
        clamp_1 = torch.clamp(
            l_self_modules_features_modules_1_modules_1_modules_attn_parameters_logit_scale_,
            max=4.605170185988092,
        )
        l_self_modules_features_modules_1_modules_1_modules_attn_parameters_logit_scale_ = (
            None
        )
        logit_scale_1 = clamp_1.exp()
        clamp_1 = None
        attn_6 = attn_5 * logit_scale_1
        attn_5 = logit_scale_1 = None
        attn_7 = attn_6 + relative_position_bias_7
        attn_6 = relative_position_bias_7 = None
        attn_mask = x_14.new_zeros((56, 56))
        x_14 = None
        attn_mask[(slice(0, -8, None), slice(0, -8, None))] = 0
        setitem = attn_mask
        setitem = None
        attn_mask[(slice(0, -8, None), slice(-8, -4, None))] = 1
        setitem_1 = attn_mask
        setitem_1 = None
        attn_mask[(slice(0, -8, None), slice(-4, None, None))] = 2
        setitem_2 = attn_mask
        setitem_2 = None
        attn_mask[(slice(-8, -4, None), slice(0, -8, None))] = 3
        setitem_3 = attn_mask
        setitem_3 = None
        attn_mask[(slice(-8, -4, None), slice(-8, -4, None))] = 4
        setitem_4 = attn_mask
        setitem_4 = None
        attn_mask[(slice(-8, -4, None), slice(-4, None, None))] = 5
        setitem_5 = attn_mask
        setitem_5 = None
        attn_mask[(slice(-4, None, None), slice(0, -8, None))] = 6
        setitem_6 = attn_mask
        setitem_6 = None
        attn_mask[(slice(-4, None, None), slice(-8, -4, None))] = 7
        setitem_7 = attn_mask
        setitem_7 = None
        attn_mask[(slice(-4, None, None), slice(-4, None, None))] = 8
        setitem_8 = attn_mask
        setitem_8 = None
        attn_mask_1 = attn_mask.view(7, 8, 7, 8)
        attn_mask = None
        permute_8 = attn_mask_1.permute(0, 2, 1, 3)
        attn_mask_1 = None
        attn_mask_2 = permute_8.reshape(49, 64)
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
        attn_8 = attn_7.view(1, 49, 3, 64, 64)
        attn_7 = None
        unsqueeze_4 = attn_mask_4.unsqueeze(1)
        attn_mask_4 = None
        unsqueeze_5 = unsqueeze_4.unsqueeze(0)
        unsqueeze_4 = None
        attn_9 = attn_8 + unsqueeze_5
        attn_8 = unsqueeze_5 = None
        attn_10 = attn_9.view(-1, 3, 64, 64)
        attn_9 = None
        attn_11 = torch.nn.functional.softmax(attn_10, dim=-1)
        attn_10 = None
        attn_12 = torch.nn.functional.dropout(attn_11, p=0.0, training=False)
        attn_11 = None
        matmul_3 = attn_12.matmul(v_1)
        attn_12 = v_1 = None
        transpose_3 = matmul_3.transpose(1, 2)
        matmul_3 = None
        x_15 = transpose_3.reshape(49, 64, 96)
        transpose_3 = None
        x_16 = torch._C._nn.linear(
            x_15,
            l_self_modules_features_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_15 = l_self_modules_features_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_17 = torch.nn.functional.dropout(x_16, p=0.0, training=False)
        x_16 = None
        x_18 = x_17.view(1, 7, 7, 8, 8, 96)
        x_17 = None
        permute_9 = x_18.permute(0, 1, 3, 2, 4, 5)
        x_18 = None
        x_19 = permute_9.reshape(1, 56, 56, 96)
        permute_9 = None
        x_20 = torch.roll(x_19, shifts=(4, 4), dims=(1, 2))
        x_19 = None
        getitem_11 = x_20[
            (
                slice(None, None, None),
                slice(None, 56, None),
                slice(None, 56, None),
                slice(None, None, None),
            )
        ]
        x_20 = None
        x_21 = getitem_11.contiguous()
        getitem_11 = None
        layer_norm_3 = torch.nn.functional.layer_norm(
            x_21,
            (96,),
            l_self_modules_features_modules_1_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_1_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_21 = (
            l_self_modules_features_modules_1_modules_1_modules_norm1_parameters_weight_
        ) = (
            l_self_modules_features_modules_1_modules_1_modules_norm1_parameters_bias_
        ) = None
        _log_api_usage_once_2 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_2 = None
        x_22 = x_10 + layer_norm_3
        x_10 = layer_norm_3 = None
        input_15 = torch._C._nn.linear(
            x_22,
            l_self_modules_features_modules_1_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_1_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_1_modules_1_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_1_modules_1_modules_mlp_modules_0_parameters_bias_ = (None)
        input_16 = torch._C._nn.gelu(input_15, approximate="none")
        input_15 = None
        input_17 = torch.nn.functional.dropout(input_16, 0.0, False, False)
        input_16 = None
        input_18 = torch._C._nn.linear(
            input_17,
            l_self_modules_features_modules_1_modules_1_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_1_modules_1_modules_mlp_modules_3_parameters_bias_,
        )
        input_17 = l_self_modules_features_modules_1_modules_1_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_1_modules_1_modules_mlp_modules_3_parameters_bias_ = (None)
        input_19 = torch.nn.functional.dropout(input_18, 0.0, False, False)
        input_18 = None
        layer_norm_4 = torch.nn.functional.layer_norm(
            input_19,
            (96,),
            l_self_modules_features_modules_1_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_1_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        input_19 = (
            l_self_modules_features_modules_1_modules_1_modules_norm2_parameters_weight_
        ) = (
            l_self_modules_features_modules_1_modules_1_modules_norm2_parameters_bias_
        ) = None
        _log_api_usage_once_3 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_3 = None
        x_23 = x_22 + layer_norm_4
        x_22 = layer_norm_4 = None
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
        x_26 = torch._C._nn.linear(
            x_25,
            l_self_modules_features_modules_2_modules_reduction_parameters_weight_,
            None,
        )
        x_25 = (
            l_self_modules_features_modules_2_modules_reduction_parameters_weight_
        ) = None
        x_27 = torch.nn.functional.layer_norm(
            x_26,
            (192,),
            l_self_modules_features_modules_2_modules_norm_parameters_weight_,
            l_self_modules_features_modules_2_modules_norm_parameters_bias_,
            1e-05,
        )
        x_26 = (
            l_self_modules_features_modules_2_modules_norm_parameters_weight_
        ) = l_self_modules_features_modules_2_modules_norm_parameters_bias_ = None
        input_20 = torch._C._nn.linear(
            l_self_modules_features_modules_3_modules_0_modules_attn_buffers_relative_coords_table_,
            l_self_modules_features_modules_3_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_3_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_3_modules_0_modules_attn_buffers_relative_coords_table_ = l_self_modules_features_modules_3_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_3_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = (None)
        input_21 = torch.nn.functional.relu(input_20, inplace=True)
        input_20 = None
        input_22 = torch._C._nn.linear(
            input_21,
            l_self_modules_features_modules_3_modules_0_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_,
            None,
        )
        input_21 = l_self_modules_features_modules_3_modules_0_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = (None)
        view_11 = input_22.view(-1, 6)
        input_22 = None
        relative_position_bias_8 = view_11[
            l_self_modules_features_modules_3_modules_0_modules_attn_buffers_relative_position_index_
        ]
        view_11 = l_self_modules_features_modules_3_modules_0_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_9 = relative_position_bias_8.view(64, 64, -1)
        relative_position_bias_8 = None
        permute_10 = relative_position_bias_9.permute(2, 0, 1)
        relative_position_bias_9 = None
        contiguous_4 = permute_10.contiguous()
        permute_10 = None
        relative_position_bias_10 = contiguous_4.unsqueeze(0)
        contiguous_4 = None
        sigmoid_2 = torch.sigmoid(relative_position_bias_10)
        relative_position_bias_10 = None
        relative_position_bias_11 = 16 * sigmoid_2
        sigmoid_2 = None
        x_28 = torch._C._nn.pad(x_27, (0, 0, 0, 4, 0, 4), "constant", None)
        x_29 = x_28.view(1, 4, 8, 4, 8, 192)
        x_28 = None
        permute_11 = x_29.permute(0, 1, 3, 2, 4, 5)
        x_29 = None
        x_30 = permute_11.reshape(16, 64, 192)
        permute_11 = None
        qkv_bias_2 = (
            l_self_modules_features_modules_3_modules_0_modules_attn_modules_qkv_parameters_bias_.clone()
        )
        l_self_modules_features_modules_3_modules_0_modules_attn_modules_qkv_parameters_bias_ = (
            None
        )
        getitem_17 = qkv_bias_2[slice(192, 384, None)]
        zero__2 = getitem_17.zero_()
        getitem_17 = zero__2 = None
        qkv_4 = torch._C._nn.linear(
            x_30,
            l_self_modules_features_modules_3_modules_0_modules_attn_modules_qkv_parameters_weight_,
            qkv_bias_2,
        )
        x_30 = l_self_modules_features_modules_3_modules_0_modules_attn_modules_qkv_parameters_weight_ = (qkv_bias_2) = (
            None
        )
        reshape_10 = qkv_4.reshape(16, 64, 3, 6, 32)
        qkv_4 = None
        qkv_5 = reshape_10.permute(2, 0, 3, 1, 4)
        reshape_10 = None
        q_2 = qkv_5[0]
        k_2 = qkv_5[1]
        v_2 = qkv_5[2]
        qkv_5 = None
        normalize_4 = torch.nn.functional.normalize(q_2, dim=-1)
        q_2 = None
        normalize_5 = torch.nn.functional.normalize(k_2, dim=-1)
        k_2 = None
        transpose_4 = normalize_5.transpose(-2, -1)
        normalize_5 = None
        attn_13 = normalize_4 @ transpose_4
        normalize_4 = transpose_4 = None
        clamp_2 = torch.clamp(
            l_self_modules_features_modules_3_modules_0_modules_attn_parameters_logit_scale_,
            max=4.605170185988092,
        )
        l_self_modules_features_modules_3_modules_0_modules_attn_parameters_logit_scale_ = (
            None
        )
        logit_scale_2 = clamp_2.exp()
        clamp_2 = None
        attn_14 = attn_13 * logit_scale_2
        attn_13 = logit_scale_2 = None
        attn_15 = attn_14 + relative_position_bias_11
        attn_14 = relative_position_bias_11 = None
        attn_16 = torch.nn.functional.softmax(attn_15, dim=-1)
        attn_15 = None
        attn_17 = torch.nn.functional.dropout(attn_16, p=0.0, training=False)
        attn_16 = None
        matmul_5 = attn_17.matmul(v_2)
        attn_17 = v_2 = None
        transpose_5 = matmul_5.transpose(1, 2)
        matmul_5 = None
        x_31 = transpose_5.reshape(16, 64, 192)
        transpose_5 = None
        x_32 = torch._C._nn.linear(
            x_31,
            l_self_modules_features_modules_3_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_3_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_31 = l_self_modules_features_modules_3_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_3_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_33 = torch.nn.functional.dropout(x_32, p=0.0, training=False)
        x_32 = None
        x_34 = x_33.view(1, 4, 4, 8, 8, 192)
        x_33 = None
        permute_13 = x_34.permute(0, 1, 3, 2, 4, 5)
        x_34 = None
        x_35 = permute_13.reshape(1, 32, 32, 192)
        permute_13 = None
        getitem_21 = x_35[
            (
                slice(None, None, None),
                slice(None, 28, None),
                slice(None, 28, None),
                slice(None, None, None),
            )
        ]
        x_35 = None
        x_36 = getitem_21.contiguous()
        getitem_21 = None
        layer_norm_6 = torch.nn.functional.layer_norm(
            x_36,
            (192,),
            l_self_modules_features_modules_3_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_3_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_36 = (
            l_self_modules_features_modules_3_modules_0_modules_norm1_parameters_weight_
        ) = (
            l_self_modules_features_modules_3_modules_0_modules_norm1_parameters_bias_
        ) = None
        _log_api_usage_once_4 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_4 = None
        x_37 = x_27 + layer_norm_6
        x_27 = layer_norm_6 = None
        input_23 = torch._C._nn.linear(
            x_37,
            l_self_modules_features_modules_3_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_3_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_3_modules_0_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_3_modules_0_modules_mlp_modules_0_parameters_bias_ = (None)
        input_24 = torch._C._nn.gelu(input_23, approximate="none")
        input_23 = None
        input_25 = torch.nn.functional.dropout(input_24, 0.0, False, False)
        input_24 = None
        input_26 = torch._C._nn.linear(
            input_25,
            l_self_modules_features_modules_3_modules_0_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_3_modules_0_modules_mlp_modules_3_parameters_bias_,
        )
        input_25 = l_self_modules_features_modules_3_modules_0_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_3_modules_0_modules_mlp_modules_3_parameters_bias_ = (None)
        input_27 = torch.nn.functional.dropout(input_26, 0.0, False, False)
        input_26 = None
        layer_norm_7 = torch.nn.functional.layer_norm(
            input_27,
            (192,),
            l_self_modules_features_modules_3_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_3_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        input_27 = (
            l_self_modules_features_modules_3_modules_0_modules_norm2_parameters_weight_
        ) = (
            l_self_modules_features_modules_3_modules_0_modules_norm2_parameters_bias_
        ) = None
        _log_api_usage_once_5 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_5 = None
        x_38 = x_37 + layer_norm_7
        x_37 = layer_norm_7 = None
        input_28 = torch._C._nn.linear(
            l_self_modules_features_modules_3_modules_1_modules_attn_buffers_relative_coords_table_,
            l_self_modules_features_modules_3_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_3_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_3_modules_1_modules_attn_buffers_relative_coords_table_ = l_self_modules_features_modules_3_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_3_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = (None)
        input_29 = torch.nn.functional.relu(input_28, inplace=True)
        input_28 = None
        input_30 = torch._C._nn.linear(
            input_29,
            l_self_modules_features_modules_3_modules_1_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_,
            None,
        )
        input_29 = l_self_modules_features_modules_3_modules_1_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = (None)
        view_15 = input_30.view(-1, 6)
        input_30 = None
        relative_position_bias_12 = view_15[
            l_self_modules_features_modules_3_modules_1_modules_attn_buffers_relative_position_index_
        ]
        view_15 = l_self_modules_features_modules_3_modules_1_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_13 = relative_position_bias_12.view(64, 64, -1)
        relative_position_bias_12 = None
        permute_14 = relative_position_bias_13.permute(2, 0, 1)
        relative_position_bias_13 = None
        contiguous_6 = permute_14.contiguous()
        permute_14 = None
        relative_position_bias_14 = contiguous_6.unsqueeze(0)
        contiguous_6 = None
        sigmoid_3 = torch.sigmoid(relative_position_bias_14)
        relative_position_bias_14 = None
        relative_position_bias_15 = 16 * sigmoid_3
        sigmoid_3 = None
        x_39 = torch._C._nn.pad(x_38, (0, 0, 0, 4, 0, 4), "constant", None)
        x_40 = torch.roll(x_39, shifts=(-4, -4), dims=(1, 2))
        x_39 = None
        x_41 = x_40.view(1, 4, 8, 4, 8, 192)
        x_40 = None
        permute_15 = x_41.permute(0, 1, 3, 2, 4, 5)
        x_41 = None
        x_42 = permute_15.reshape(16, 64, 192)
        permute_15 = None
        qkv_bias_3 = (
            l_self_modules_features_modules_3_modules_1_modules_attn_modules_qkv_parameters_bias_.clone()
        )
        l_self_modules_features_modules_3_modules_1_modules_attn_modules_qkv_parameters_bias_ = (
            None
        )
        getitem_23 = qkv_bias_3[slice(192, 384, None)]
        zero__3 = getitem_23.zero_()
        getitem_23 = zero__3 = None
        qkv_6 = torch._C._nn.linear(
            x_42,
            l_self_modules_features_modules_3_modules_1_modules_attn_modules_qkv_parameters_weight_,
            qkv_bias_3,
        )
        l_self_modules_features_modules_3_modules_1_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_3
        ) = None
        reshape_14 = qkv_6.reshape(16, 64, 3, 6, 32)
        qkv_6 = None
        qkv_7 = reshape_14.permute(2, 0, 3, 1, 4)
        reshape_14 = None
        q_3 = qkv_7[0]
        k_3 = qkv_7[1]
        v_3 = qkv_7[2]
        qkv_7 = None
        normalize_6 = torch.nn.functional.normalize(q_3, dim=-1)
        q_3 = None
        normalize_7 = torch.nn.functional.normalize(k_3, dim=-1)
        k_3 = None
        transpose_6 = normalize_7.transpose(-2, -1)
        normalize_7 = None
        attn_18 = normalize_6 @ transpose_6
        normalize_6 = transpose_6 = None
        clamp_3 = torch.clamp(
            l_self_modules_features_modules_3_modules_1_modules_attn_parameters_logit_scale_,
            max=4.605170185988092,
        )
        l_self_modules_features_modules_3_modules_1_modules_attn_parameters_logit_scale_ = (
            None
        )
        logit_scale_3 = clamp_3.exp()
        clamp_3 = None
        attn_19 = attn_18 * logit_scale_3
        attn_18 = logit_scale_3 = None
        attn_20 = attn_19 + relative_position_bias_15
        attn_19 = relative_position_bias_15 = None
        attn_mask_5 = x_42.new_zeros((32, 32))
        x_42 = None
        attn_mask_5[(slice(0, -8, None), slice(0, -8, None))] = 0
        setitem_9 = attn_mask_5
        setitem_9 = None
        attn_mask_5[(slice(0, -8, None), slice(-8, -4, None))] = 1
        setitem_10 = attn_mask_5
        setitem_10 = None
        attn_mask_5[(slice(0, -8, None), slice(-4, None, None))] = 2
        setitem_11 = attn_mask_5
        setitem_11 = None
        attn_mask_5[(slice(-8, -4, None), slice(0, -8, None))] = 3
        setitem_12 = attn_mask_5
        setitem_12 = None
        attn_mask_5[(slice(-8, -4, None), slice(-8, -4, None))] = 4
        setitem_13 = attn_mask_5
        setitem_13 = None
        attn_mask_5[(slice(-8, -4, None), slice(-4, None, None))] = 5
        setitem_14 = attn_mask_5
        setitem_14 = None
        attn_mask_5[(slice(-4, None, None), slice(0, -8, None))] = 6
        setitem_15 = attn_mask_5
        setitem_15 = None
        attn_mask_5[(slice(-4, None, None), slice(-8, -4, None))] = 7
        setitem_16 = attn_mask_5
        setitem_16 = None
        attn_mask_5[(slice(-4, None, None), slice(-4, None, None))] = 8
        setitem_17 = attn_mask_5
        setitem_17 = None
        attn_mask_6 = attn_mask_5.view(4, 8, 4, 8)
        attn_mask_5 = None
        permute_17 = attn_mask_6.permute(0, 2, 1, 3)
        attn_mask_6 = None
        attn_mask_7 = permute_17.reshape(16, 64)
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
        attn_21 = attn_20.view(1, 16, 6, 64, 64)
        attn_20 = None
        unsqueeze_10 = attn_mask_9.unsqueeze(1)
        attn_mask_9 = None
        unsqueeze_11 = unsqueeze_10.unsqueeze(0)
        unsqueeze_10 = None
        attn_22 = attn_21 + unsqueeze_11
        attn_21 = unsqueeze_11 = None
        attn_23 = attn_22.view(-1, 6, 64, 64)
        attn_22 = None
        attn_24 = torch.nn.functional.softmax(attn_23, dim=-1)
        attn_23 = None
        attn_25 = torch.nn.functional.dropout(attn_24, p=0.0, training=False)
        attn_24 = None
        matmul_7 = attn_25.matmul(v_3)
        attn_25 = v_3 = None
        transpose_7 = matmul_7.transpose(1, 2)
        matmul_7 = None
        x_43 = transpose_7.reshape(16, 64, 192)
        transpose_7 = None
        x_44 = torch._C._nn.linear(
            x_43,
            l_self_modules_features_modules_3_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_3_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_43 = l_self_modules_features_modules_3_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_3_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_45 = torch.nn.functional.dropout(x_44, p=0.0, training=False)
        x_44 = None
        x_46 = x_45.view(1, 4, 4, 8, 8, 192)
        x_45 = None
        permute_18 = x_46.permute(0, 1, 3, 2, 4, 5)
        x_46 = None
        x_47 = permute_18.reshape(1, 32, 32, 192)
        permute_18 = None
        x_48 = torch.roll(x_47, shifts=(4, 4), dims=(1, 2))
        x_47 = None
        getitem_27 = x_48[
            (
                slice(None, None, None),
                slice(None, 28, None),
                slice(None, 28, None),
                slice(None, None, None),
            )
        ]
        x_48 = None
        x_49 = getitem_27.contiguous()
        getitem_27 = None
        layer_norm_8 = torch.nn.functional.layer_norm(
            x_49,
            (192,),
            l_self_modules_features_modules_3_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_3_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_49 = (
            l_self_modules_features_modules_3_modules_1_modules_norm1_parameters_weight_
        ) = (
            l_self_modules_features_modules_3_modules_1_modules_norm1_parameters_bias_
        ) = None
        _log_api_usage_once_6 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_6 = None
        x_50 = x_38 + layer_norm_8
        x_38 = layer_norm_8 = None
        input_31 = torch._C._nn.linear(
            x_50,
            l_self_modules_features_modules_3_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_3_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_3_modules_1_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_3_modules_1_modules_mlp_modules_0_parameters_bias_ = (None)
        input_32 = torch._C._nn.gelu(input_31, approximate="none")
        input_31 = None
        input_33 = torch.nn.functional.dropout(input_32, 0.0, False, False)
        input_32 = None
        input_34 = torch._C._nn.linear(
            input_33,
            l_self_modules_features_modules_3_modules_1_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_3_modules_1_modules_mlp_modules_3_parameters_bias_,
        )
        input_33 = l_self_modules_features_modules_3_modules_1_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_3_modules_1_modules_mlp_modules_3_parameters_bias_ = (None)
        input_35 = torch.nn.functional.dropout(input_34, 0.0, False, False)
        input_34 = None
        layer_norm_9 = torch.nn.functional.layer_norm(
            input_35,
            (192,),
            l_self_modules_features_modules_3_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_3_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        input_35 = (
            l_self_modules_features_modules_3_modules_1_modules_norm2_parameters_weight_
        ) = (
            l_self_modules_features_modules_3_modules_1_modules_norm2_parameters_bias_
        ) = None
        _log_api_usage_once_7 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_7 = None
        x_51 = x_50 + layer_norm_9
        x_50 = layer_norm_9 = None
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
        x_54 = torch._C._nn.linear(
            x_53,
            l_self_modules_features_modules_4_modules_reduction_parameters_weight_,
            None,
        )
        x_53 = (
            l_self_modules_features_modules_4_modules_reduction_parameters_weight_
        ) = None
        x_55 = torch.nn.functional.layer_norm(
            x_54,
            (384,),
            l_self_modules_features_modules_4_modules_norm_parameters_weight_,
            l_self_modules_features_modules_4_modules_norm_parameters_bias_,
            1e-05,
        )
        x_54 = (
            l_self_modules_features_modules_4_modules_norm_parameters_weight_
        ) = l_self_modules_features_modules_4_modules_norm_parameters_bias_ = None
        input_36 = torch._C._nn.linear(
            l_self_modules_features_modules_5_modules_0_modules_attn_buffers_relative_coords_table_,
            l_self_modules_features_modules_5_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_0_modules_attn_buffers_relative_coords_table_ = l_self_modules_features_modules_5_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = (None)
        input_37 = torch.nn.functional.relu(input_36, inplace=True)
        input_36 = None
        input_38 = torch._C._nn.linear(
            input_37,
            l_self_modules_features_modules_5_modules_0_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_,
            None,
        )
        input_37 = l_self_modules_features_modules_5_modules_0_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = (None)
        view_22 = input_38.view(-1, 12)
        input_38 = None
        relative_position_bias_16 = view_22[
            l_self_modules_features_modules_5_modules_0_modules_attn_buffers_relative_position_index_
        ]
        view_22 = l_self_modules_features_modules_5_modules_0_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_17 = relative_position_bias_16.view(64, 64, -1)
        relative_position_bias_16 = None
        permute_19 = relative_position_bias_17.permute(2, 0, 1)
        relative_position_bias_17 = None
        contiguous_8 = permute_19.contiguous()
        permute_19 = None
        relative_position_bias_18 = contiguous_8.unsqueeze(0)
        contiguous_8 = None
        sigmoid_4 = torch.sigmoid(relative_position_bias_18)
        relative_position_bias_18 = None
        relative_position_bias_19 = 16 * sigmoid_4
        sigmoid_4 = None
        x_56 = torch._C._nn.pad(x_55, (0, 0, 0, 2, 0, 2), "constant", None)
        x_57 = x_56.view(1, 2, 8, 2, 8, 384)
        x_56 = None
        permute_20 = x_57.permute(0, 1, 3, 2, 4, 5)
        x_57 = None
        x_58 = permute_20.reshape(4, 64, 384)
        permute_20 = None
        qkv_bias_4 = (
            l_self_modules_features_modules_5_modules_0_modules_attn_modules_qkv_parameters_bias_.clone()
        )
        l_self_modules_features_modules_5_modules_0_modules_attn_modules_qkv_parameters_bias_ = (
            None
        )
        getitem_33 = qkv_bias_4[slice(384, 768, None)]
        zero__4 = getitem_33.zero_()
        getitem_33 = zero__4 = None
        qkv_8 = torch._C._nn.linear(
            x_58,
            l_self_modules_features_modules_5_modules_0_modules_attn_modules_qkv_parameters_weight_,
            qkv_bias_4,
        )
        x_58 = l_self_modules_features_modules_5_modules_0_modules_attn_modules_qkv_parameters_weight_ = (qkv_bias_4) = (
            None
        )
        reshape_19 = qkv_8.reshape(4, 64, 3, 12, 32)
        qkv_8 = None
        qkv_9 = reshape_19.permute(2, 0, 3, 1, 4)
        reshape_19 = None
        q_4 = qkv_9[0]
        k_4 = qkv_9[1]
        v_4 = qkv_9[2]
        qkv_9 = None
        normalize_8 = torch.nn.functional.normalize(q_4, dim=-1)
        q_4 = None
        normalize_9 = torch.nn.functional.normalize(k_4, dim=-1)
        k_4 = None
        transpose_8 = normalize_9.transpose(-2, -1)
        normalize_9 = None
        attn_26 = normalize_8 @ transpose_8
        normalize_8 = transpose_8 = None
        clamp_4 = torch.clamp(
            l_self_modules_features_modules_5_modules_0_modules_attn_parameters_logit_scale_,
            max=4.605170185988092,
        )
        l_self_modules_features_modules_5_modules_0_modules_attn_parameters_logit_scale_ = (
            None
        )
        logit_scale_4 = clamp_4.exp()
        clamp_4 = None
        attn_27 = attn_26 * logit_scale_4
        attn_26 = logit_scale_4 = None
        attn_28 = attn_27 + relative_position_bias_19
        attn_27 = relative_position_bias_19 = None
        attn_29 = torch.nn.functional.softmax(attn_28, dim=-1)
        attn_28 = None
        attn_30 = torch.nn.functional.dropout(attn_29, p=0.0, training=False)
        attn_29 = None
        matmul_9 = attn_30.matmul(v_4)
        attn_30 = v_4 = None
        transpose_9 = matmul_9.transpose(1, 2)
        matmul_9 = None
        x_59 = transpose_9.reshape(4, 64, 384)
        transpose_9 = None
        x_60 = torch._C._nn.linear(
            x_59,
            l_self_modules_features_modules_5_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_59 = l_self_modules_features_modules_5_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_61 = torch.nn.functional.dropout(x_60, p=0.0, training=False)
        x_60 = None
        x_62 = x_61.view(1, 2, 2, 8, 8, 384)
        x_61 = None
        permute_22 = x_62.permute(0, 1, 3, 2, 4, 5)
        x_62 = None
        x_63 = permute_22.reshape(1, 16, 16, 384)
        permute_22 = None
        getitem_37 = x_63[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_63 = None
        x_64 = getitem_37.contiguous()
        getitem_37 = None
        layer_norm_11 = torch.nn.functional.layer_norm(
            x_64,
            (384,),
            l_self_modules_features_modules_5_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_64 = (
            l_self_modules_features_modules_5_modules_0_modules_norm1_parameters_weight_
        ) = (
            l_self_modules_features_modules_5_modules_0_modules_norm1_parameters_bias_
        ) = None
        _log_api_usage_once_8 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_8 = None
        x_65 = x_55 + layer_norm_11
        x_55 = layer_norm_11 = None
        input_39 = torch._C._nn.linear(
            x_65,
            l_self_modules_features_modules_5_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_0_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_mlp_modules_0_parameters_bias_ = (None)
        input_40 = torch._C._nn.gelu(input_39, approximate="none")
        input_39 = None
        input_41 = torch.nn.functional.dropout(input_40, 0.0, False, False)
        input_40 = None
        input_42 = torch._C._nn.linear(
            input_41,
            l_self_modules_features_modules_5_modules_0_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_mlp_modules_3_parameters_bias_,
        )
        input_41 = l_self_modules_features_modules_5_modules_0_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_mlp_modules_3_parameters_bias_ = (None)
        input_43 = torch.nn.functional.dropout(input_42, 0.0, False, False)
        input_42 = None
        layer_norm_12 = torch.nn.functional.layer_norm(
            input_43,
            (384,),
            l_self_modules_features_modules_5_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        input_43 = (
            l_self_modules_features_modules_5_modules_0_modules_norm2_parameters_weight_
        ) = (
            l_self_modules_features_modules_5_modules_0_modules_norm2_parameters_bias_
        ) = None
        _log_api_usage_once_9 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_9 = None
        x_66 = x_65 + layer_norm_12
        x_65 = layer_norm_12 = None
        input_44 = torch._C._nn.linear(
            l_self_modules_features_modules_5_modules_1_modules_attn_buffers_relative_coords_table_,
            l_self_modules_features_modules_5_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_1_modules_attn_buffers_relative_coords_table_ = l_self_modules_features_modules_5_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = (None)
        input_45 = torch.nn.functional.relu(input_44, inplace=True)
        input_44 = None
        input_46 = torch._C._nn.linear(
            input_45,
            l_self_modules_features_modules_5_modules_1_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_,
            None,
        )
        input_45 = l_self_modules_features_modules_5_modules_1_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = (None)
        view_26 = input_46.view(-1, 12)
        input_46 = None
        relative_position_bias_20 = view_26[
            l_self_modules_features_modules_5_modules_1_modules_attn_buffers_relative_position_index_
        ]
        view_26 = l_self_modules_features_modules_5_modules_1_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_21 = relative_position_bias_20.view(64, 64, -1)
        relative_position_bias_20 = None
        permute_23 = relative_position_bias_21.permute(2, 0, 1)
        relative_position_bias_21 = None
        contiguous_10 = permute_23.contiguous()
        permute_23 = None
        relative_position_bias_22 = contiguous_10.unsqueeze(0)
        contiguous_10 = None
        sigmoid_5 = torch.sigmoid(relative_position_bias_22)
        relative_position_bias_22 = None
        relative_position_bias_23 = 16 * sigmoid_5
        sigmoid_5 = None
        x_67 = torch._C._nn.pad(x_66, (0, 0, 0, 2, 0, 2), "constant", None)
        x_68 = torch.roll(x_67, shifts=(-4, -4), dims=(1, 2))
        x_67 = None
        x_69 = x_68.view(1, 2, 8, 2, 8, 384)
        x_68 = None
        permute_24 = x_69.permute(0, 1, 3, 2, 4, 5)
        x_69 = None
        x_70 = permute_24.reshape(4, 64, 384)
        permute_24 = None
        qkv_bias_5 = (
            l_self_modules_features_modules_5_modules_1_modules_attn_modules_qkv_parameters_bias_.clone()
        )
        l_self_modules_features_modules_5_modules_1_modules_attn_modules_qkv_parameters_bias_ = (
            None
        )
        getitem_39 = qkv_bias_5[slice(384, 768, None)]
        zero__5 = getitem_39.zero_()
        getitem_39 = zero__5 = None
        qkv_10 = torch._C._nn.linear(
            x_70,
            l_self_modules_features_modules_5_modules_1_modules_attn_modules_qkv_parameters_weight_,
            qkv_bias_5,
        )
        l_self_modules_features_modules_5_modules_1_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_5
        ) = None
        reshape_23 = qkv_10.reshape(4, 64, 3, 12, 32)
        qkv_10 = None
        qkv_11 = reshape_23.permute(2, 0, 3, 1, 4)
        reshape_23 = None
        q_5 = qkv_11[0]
        k_5 = qkv_11[1]
        v_5 = qkv_11[2]
        qkv_11 = None
        normalize_10 = torch.nn.functional.normalize(q_5, dim=-1)
        q_5 = None
        normalize_11 = torch.nn.functional.normalize(k_5, dim=-1)
        k_5 = None
        transpose_10 = normalize_11.transpose(-2, -1)
        normalize_11 = None
        attn_31 = normalize_10 @ transpose_10
        normalize_10 = transpose_10 = None
        clamp_5 = torch.clamp(
            l_self_modules_features_modules_5_modules_1_modules_attn_parameters_logit_scale_,
            max=4.605170185988092,
        )
        l_self_modules_features_modules_5_modules_1_modules_attn_parameters_logit_scale_ = (
            None
        )
        logit_scale_5 = clamp_5.exp()
        clamp_5 = None
        attn_32 = attn_31 * logit_scale_5
        attn_31 = logit_scale_5 = None
        attn_33 = attn_32 + relative_position_bias_23
        attn_32 = relative_position_bias_23 = None
        attn_mask_10 = x_70.new_zeros((16, 16))
        x_70 = None
        attn_mask_10[(slice(0, -8, None), slice(0, -8, None))] = 0
        setitem_18 = attn_mask_10
        setitem_18 = None
        attn_mask_10[(slice(0, -8, None), slice(-8, -4, None))] = 1
        setitem_19 = attn_mask_10
        setitem_19 = None
        attn_mask_10[(slice(0, -8, None), slice(-4, None, None))] = 2
        setitem_20 = attn_mask_10
        setitem_20 = None
        attn_mask_10[(slice(-8, -4, None), slice(0, -8, None))] = 3
        setitem_21 = attn_mask_10
        setitem_21 = None
        attn_mask_10[(slice(-8, -4, None), slice(-8, -4, None))] = 4
        setitem_22 = attn_mask_10
        setitem_22 = None
        attn_mask_10[(slice(-8, -4, None), slice(-4, None, None))] = 5
        setitem_23 = attn_mask_10
        setitem_23 = None
        attn_mask_10[(slice(-4, None, None), slice(0, -8, None))] = 6
        setitem_24 = attn_mask_10
        setitem_24 = None
        attn_mask_10[(slice(-4, None, None), slice(-8, -4, None))] = 7
        setitem_25 = attn_mask_10
        setitem_25 = None
        attn_mask_10[(slice(-4, None, None), slice(-4, None, None))] = 8
        setitem_26 = attn_mask_10
        setitem_26 = None
        attn_mask_11 = attn_mask_10.view(2, 8, 2, 8)
        attn_mask_10 = None
        permute_26 = attn_mask_11.permute(0, 2, 1, 3)
        attn_mask_11 = None
        attn_mask_12 = permute_26.reshape(4, 64)
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
        attn_34 = attn_33.view(1, 4, 12, 64, 64)
        attn_33 = None
        unsqueeze_16 = attn_mask_14.unsqueeze(1)
        attn_mask_14 = None
        unsqueeze_17 = unsqueeze_16.unsqueeze(0)
        unsqueeze_16 = None
        attn_35 = attn_34 + unsqueeze_17
        attn_34 = unsqueeze_17 = None
        attn_36 = attn_35.view(-1, 12, 64, 64)
        attn_35 = None
        attn_37 = torch.nn.functional.softmax(attn_36, dim=-1)
        attn_36 = None
        attn_38 = torch.nn.functional.dropout(attn_37, p=0.0, training=False)
        attn_37 = None
        matmul_11 = attn_38.matmul(v_5)
        attn_38 = v_5 = None
        transpose_11 = matmul_11.transpose(1, 2)
        matmul_11 = None
        x_71 = transpose_11.reshape(4, 64, 384)
        transpose_11 = None
        x_72 = torch._C._nn.linear(
            x_71,
            l_self_modules_features_modules_5_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_71 = l_self_modules_features_modules_5_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_73 = torch.nn.functional.dropout(x_72, p=0.0, training=False)
        x_72 = None
        x_74 = x_73.view(1, 2, 2, 8, 8, 384)
        x_73 = None
        permute_27 = x_74.permute(0, 1, 3, 2, 4, 5)
        x_74 = None
        x_75 = permute_27.reshape(1, 16, 16, 384)
        permute_27 = None
        x_76 = torch.roll(x_75, shifts=(4, 4), dims=(1, 2))
        x_75 = None
        getitem_43 = x_76[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_76 = None
        x_77 = getitem_43.contiguous()
        getitem_43 = None
        layer_norm_13 = torch.nn.functional.layer_norm(
            x_77,
            (384,),
            l_self_modules_features_modules_5_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_77 = (
            l_self_modules_features_modules_5_modules_1_modules_norm1_parameters_weight_
        ) = (
            l_self_modules_features_modules_5_modules_1_modules_norm1_parameters_bias_
        ) = None
        _log_api_usage_once_10 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_10 = None
        x_78 = x_66 + layer_norm_13
        x_66 = layer_norm_13 = None
        input_47 = torch._C._nn.linear(
            x_78,
            l_self_modules_features_modules_5_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_1_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_mlp_modules_0_parameters_bias_ = (None)
        input_48 = torch._C._nn.gelu(input_47, approximate="none")
        input_47 = None
        input_49 = torch.nn.functional.dropout(input_48, 0.0, False, False)
        input_48 = None
        input_50 = torch._C._nn.linear(
            input_49,
            l_self_modules_features_modules_5_modules_1_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_mlp_modules_3_parameters_bias_,
        )
        input_49 = l_self_modules_features_modules_5_modules_1_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_mlp_modules_3_parameters_bias_ = (None)
        input_51 = torch.nn.functional.dropout(input_50, 0.0, False, False)
        input_50 = None
        layer_norm_14 = torch.nn.functional.layer_norm(
            input_51,
            (384,),
            l_self_modules_features_modules_5_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        input_51 = (
            l_self_modules_features_modules_5_modules_1_modules_norm2_parameters_weight_
        ) = (
            l_self_modules_features_modules_5_modules_1_modules_norm2_parameters_bias_
        ) = None
        _log_api_usage_once_11 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_11 = None
        x_79 = x_78 + layer_norm_14
        x_78 = layer_norm_14 = None
        input_52 = torch._C._nn.linear(
            l_self_modules_features_modules_5_modules_2_modules_attn_buffers_relative_coords_table_,
            l_self_modules_features_modules_5_modules_2_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_2_modules_attn_buffers_relative_coords_table_ = l_self_modules_features_modules_5_modules_2_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = (None)
        input_53 = torch.nn.functional.relu(input_52, inplace=True)
        input_52 = None
        input_54 = torch._C._nn.linear(
            input_53,
            l_self_modules_features_modules_5_modules_2_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_,
            None,
        )
        input_53 = l_self_modules_features_modules_5_modules_2_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = (None)
        view_33 = input_54.view(-1, 12)
        input_54 = None
        relative_position_bias_24 = view_33[
            l_self_modules_features_modules_5_modules_2_modules_attn_buffers_relative_position_index_
        ]
        view_33 = l_self_modules_features_modules_5_modules_2_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_25 = relative_position_bias_24.view(64, 64, -1)
        relative_position_bias_24 = None
        permute_28 = relative_position_bias_25.permute(2, 0, 1)
        relative_position_bias_25 = None
        contiguous_12 = permute_28.contiguous()
        permute_28 = None
        relative_position_bias_26 = contiguous_12.unsqueeze(0)
        contiguous_12 = None
        sigmoid_6 = torch.sigmoid(relative_position_bias_26)
        relative_position_bias_26 = None
        relative_position_bias_27 = 16 * sigmoid_6
        sigmoid_6 = None
        x_80 = torch._C._nn.pad(x_79, (0, 0, 0, 2, 0, 2), "constant", None)
        x_81 = x_80.view(1, 2, 8, 2, 8, 384)
        x_80 = None
        permute_29 = x_81.permute(0, 1, 3, 2, 4, 5)
        x_81 = None
        x_82 = permute_29.reshape(4, 64, 384)
        permute_29 = None
        qkv_bias_6 = (
            l_self_modules_features_modules_5_modules_2_modules_attn_modules_qkv_parameters_bias_.clone()
        )
        l_self_modules_features_modules_5_modules_2_modules_attn_modules_qkv_parameters_bias_ = (
            None
        )
        getitem_45 = qkv_bias_6[slice(384, 768, None)]
        zero__6 = getitem_45.zero_()
        getitem_45 = zero__6 = None
        qkv_12 = torch._C._nn.linear(
            x_82,
            l_self_modules_features_modules_5_modules_2_modules_attn_modules_qkv_parameters_weight_,
            qkv_bias_6,
        )
        x_82 = l_self_modules_features_modules_5_modules_2_modules_attn_modules_qkv_parameters_weight_ = (qkv_bias_6) = (
            None
        )
        reshape_28 = qkv_12.reshape(4, 64, 3, 12, 32)
        qkv_12 = None
        qkv_13 = reshape_28.permute(2, 0, 3, 1, 4)
        reshape_28 = None
        q_6 = qkv_13[0]
        k_6 = qkv_13[1]
        v_6 = qkv_13[2]
        qkv_13 = None
        normalize_12 = torch.nn.functional.normalize(q_6, dim=-1)
        q_6 = None
        normalize_13 = torch.nn.functional.normalize(k_6, dim=-1)
        k_6 = None
        transpose_12 = normalize_13.transpose(-2, -1)
        normalize_13 = None
        attn_39 = normalize_12 @ transpose_12
        normalize_12 = transpose_12 = None
        clamp_6 = torch.clamp(
            l_self_modules_features_modules_5_modules_2_modules_attn_parameters_logit_scale_,
            max=4.605170185988092,
        )
        l_self_modules_features_modules_5_modules_2_modules_attn_parameters_logit_scale_ = (
            None
        )
        logit_scale_6 = clamp_6.exp()
        clamp_6 = None
        attn_40 = attn_39 * logit_scale_6
        attn_39 = logit_scale_6 = None
        attn_41 = attn_40 + relative_position_bias_27
        attn_40 = relative_position_bias_27 = None
        attn_42 = torch.nn.functional.softmax(attn_41, dim=-1)
        attn_41 = None
        attn_43 = torch.nn.functional.dropout(attn_42, p=0.0, training=False)
        attn_42 = None
        matmul_13 = attn_43.matmul(v_6)
        attn_43 = v_6 = None
        transpose_13 = matmul_13.transpose(1, 2)
        matmul_13 = None
        x_83 = transpose_13.reshape(4, 64, 384)
        transpose_13 = None
        x_84 = torch._C._nn.linear(
            x_83,
            l_self_modules_features_modules_5_modules_2_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_attn_modules_proj_parameters_bias_,
        )
        x_83 = l_self_modules_features_modules_5_modules_2_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_attn_modules_proj_parameters_bias_ = (None)
        x_85 = torch.nn.functional.dropout(x_84, p=0.0, training=False)
        x_84 = None
        x_86 = x_85.view(1, 2, 2, 8, 8, 384)
        x_85 = None
        permute_31 = x_86.permute(0, 1, 3, 2, 4, 5)
        x_86 = None
        x_87 = permute_31.reshape(1, 16, 16, 384)
        permute_31 = None
        getitem_49 = x_87[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_87 = None
        x_88 = getitem_49.contiguous()
        getitem_49 = None
        layer_norm_15 = torch.nn.functional.layer_norm(
            x_88,
            (384,),
            l_self_modules_features_modules_5_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_88 = (
            l_self_modules_features_modules_5_modules_2_modules_norm1_parameters_weight_
        ) = (
            l_self_modules_features_modules_5_modules_2_modules_norm1_parameters_bias_
        ) = None
        _log_api_usage_once_12 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_12 = None
        x_89 = x_79 + layer_norm_15
        x_79 = layer_norm_15 = None
        input_55 = torch._C._nn.linear(
            x_89,
            l_self_modules_features_modules_5_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_2_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_mlp_modules_0_parameters_bias_ = (None)
        input_56 = torch._C._nn.gelu(input_55, approximate="none")
        input_55 = None
        input_57 = torch.nn.functional.dropout(input_56, 0.0, False, False)
        input_56 = None
        input_58 = torch._C._nn.linear(
            input_57,
            l_self_modules_features_modules_5_modules_2_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_mlp_modules_3_parameters_bias_,
        )
        input_57 = l_self_modules_features_modules_5_modules_2_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_mlp_modules_3_parameters_bias_ = (None)
        input_59 = torch.nn.functional.dropout(input_58, 0.0, False, False)
        input_58 = None
        layer_norm_16 = torch.nn.functional.layer_norm(
            input_59,
            (384,),
            l_self_modules_features_modules_5_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_norm2_parameters_bias_,
            1e-05,
        )
        input_59 = (
            l_self_modules_features_modules_5_modules_2_modules_norm2_parameters_weight_
        ) = (
            l_self_modules_features_modules_5_modules_2_modules_norm2_parameters_bias_
        ) = None
        _log_api_usage_once_13 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_13 = None
        x_90 = x_89 + layer_norm_16
        x_89 = layer_norm_16 = None
        input_60 = torch._C._nn.linear(
            l_self_modules_features_modules_5_modules_3_modules_attn_buffers_relative_coords_table_,
            l_self_modules_features_modules_5_modules_3_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_3_modules_attn_buffers_relative_coords_table_ = l_self_modules_features_modules_5_modules_3_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = (None)
        input_61 = torch.nn.functional.relu(input_60, inplace=True)
        input_60 = None
        input_62 = torch._C._nn.linear(
            input_61,
            l_self_modules_features_modules_5_modules_3_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_,
            None,
        )
        input_61 = l_self_modules_features_modules_5_modules_3_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = (None)
        view_37 = input_62.view(-1, 12)
        input_62 = None
        relative_position_bias_28 = view_37[
            l_self_modules_features_modules_5_modules_3_modules_attn_buffers_relative_position_index_
        ]
        view_37 = l_self_modules_features_modules_5_modules_3_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_29 = relative_position_bias_28.view(64, 64, -1)
        relative_position_bias_28 = None
        permute_32 = relative_position_bias_29.permute(2, 0, 1)
        relative_position_bias_29 = None
        contiguous_14 = permute_32.contiguous()
        permute_32 = None
        relative_position_bias_30 = contiguous_14.unsqueeze(0)
        contiguous_14 = None
        sigmoid_7 = torch.sigmoid(relative_position_bias_30)
        relative_position_bias_30 = None
        relative_position_bias_31 = 16 * sigmoid_7
        sigmoid_7 = None
        x_91 = torch._C._nn.pad(x_90, (0, 0, 0, 2, 0, 2), "constant", None)
        x_92 = torch.roll(x_91, shifts=(-4, -4), dims=(1, 2))
        x_91 = None
        x_93 = x_92.view(1, 2, 8, 2, 8, 384)
        x_92 = None
        permute_33 = x_93.permute(0, 1, 3, 2, 4, 5)
        x_93 = None
        x_94 = permute_33.reshape(4, 64, 384)
        permute_33 = None
        qkv_bias_7 = (
            l_self_modules_features_modules_5_modules_3_modules_attn_modules_qkv_parameters_bias_.clone()
        )
        l_self_modules_features_modules_5_modules_3_modules_attn_modules_qkv_parameters_bias_ = (
            None
        )
        getitem_51 = qkv_bias_7[slice(384, 768, None)]
        zero__7 = getitem_51.zero_()
        getitem_51 = zero__7 = None
        qkv_14 = torch._C._nn.linear(
            x_94,
            l_self_modules_features_modules_5_modules_3_modules_attn_modules_qkv_parameters_weight_,
            qkv_bias_7,
        )
        l_self_modules_features_modules_5_modules_3_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_7
        ) = None
        reshape_32 = qkv_14.reshape(4, 64, 3, 12, 32)
        qkv_14 = None
        qkv_15 = reshape_32.permute(2, 0, 3, 1, 4)
        reshape_32 = None
        q_7 = qkv_15[0]
        k_7 = qkv_15[1]
        v_7 = qkv_15[2]
        qkv_15 = None
        normalize_14 = torch.nn.functional.normalize(q_7, dim=-1)
        q_7 = None
        normalize_15 = torch.nn.functional.normalize(k_7, dim=-1)
        k_7 = None
        transpose_14 = normalize_15.transpose(-2, -1)
        normalize_15 = None
        attn_44 = normalize_14 @ transpose_14
        normalize_14 = transpose_14 = None
        clamp_7 = torch.clamp(
            l_self_modules_features_modules_5_modules_3_modules_attn_parameters_logit_scale_,
            max=4.605170185988092,
        )
        l_self_modules_features_modules_5_modules_3_modules_attn_parameters_logit_scale_ = (
            None
        )
        logit_scale_7 = clamp_7.exp()
        clamp_7 = None
        attn_45 = attn_44 * logit_scale_7
        attn_44 = logit_scale_7 = None
        attn_46 = attn_45 + relative_position_bias_31
        attn_45 = relative_position_bias_31 = None
        attn_mask_15 = x_94.new_zeros((16, 16))
        x_94 = None
        attn_mask_15[(slice(0, -8, None), slice(0, -8, None))] = 0
        setitem_27 = attn_mask_15
        setitem_27 = None
        attn_mask_15[(slice(0, -8, None), slice(-8, -4, None))] = 1
        setitem_28 = attn_mask_15
        setitem_28 = None
        attn_mask_15[(slice(0, -8, None), slice(-4, None, None))] = 2
        setitem_29 = attn_mask_15
        setitem_29 = None
        attn_mask_15[(slice(-8, -4, None), slice(0, -8, None))] = 3
        setitem_30 = attn_mask_15
        setitem_30 = None
        attn_mask_15[(slice(-8, -4, None), slice(-8, -4, None))] = 4
        setitem_31 = attn_mask_15
        setitem_31 = None
        attn_mask_15[(slice(-8, -4, None), slice(-4, None, None))] = 5
        setitem_32 = attn_mask_15
        setitem_32 = None
        attn_mask_15[(slice(-4, None, None), slice(0, -8, None))] = 6
        setitem_33 = attn_mask_15
        setitem_33 = None
        attn_mask_15[(slice(-4, None, None), slice(-8, -4, None))] = 7
        setitem_34 = attn_mask_15
        setitem_34 = None
        attn_mask_15[(slice(-4, None, None), slice(-4, None, None))] = 8
        setitem_35 = attn_mask_15
        setitem_35 = None
        attn_mask_16 = attn_mask_15.view(2, 8, 2, 8)
        attn_mask_15 = None
        permute_35 = attn_mask_16.permute(0, 2, 1, 3)
        attn_mask_16 = None
        attn_mask_17 = permute_35.reshape(4, 64)
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
        attn_47 = attn_46.view(1, 4, 12, 64, 64)
        attn_46 = None
        unsqueeze_22 = attn_mask_19.unsqueeze(1)
        attn_mask_19 = None
        unsqueeze_23 = unsqueeze_22.unsqueeze(0)
        unsqueeze_22 = None
        attn_48 = attn_47 + unsqueeze_23
        attn_47 = unsqueeze_23 = None
        attn_49 = attn_48.view(-1, 12, 64, 64)
        attn_48 = None
        attn_50 = torch.nn.functional.softmax(attn_49, dim=-1)
        attn_49 = None
        attn_51 = torch.nn.functional.dropout(attn_50, p=0.0, training=False)
        attn_50 = None
        matmul_15 = attn_51.matmul(v_7)
        attn_51 = v_7 = None
        transpose_15 = matmul_15.transpose(1, 2)
        matmul_15 = None
        x_95 = transpose_15.reshape(4, 64, 384)
        transpose_15 = None
        x_96 = torch._C._nn.linear(
            x_95,
            l_self_modules_features_modules_5_modules_3_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_attn_modules_proj_parameters_bias_,
        )
        x_95 = l_self_modules_features_modules_5_modules_3_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_attn_modules_proj_parameters_bias_ = (None)
        x_97 = torch.nn.functional.dropout(x_96, p=0.0, training=False)
        x_96 = None
        x_98 = x_97.view(1, 2, 2, 8, 8, 384)
        x_97 = None
        permute_36 = x_98.permute(0, 1, 3, 2, 4, 5)
        x_98 = None
        x_99 = permute_36.reshape(1, 16, 16, 384)
        permute_36 = None
        x_100 = torch.roll(x_99, shifts=(4, 4), dims=(1, 2))
        x_99 = None
        getitem_55 = x_100[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_100 = None
        x_101 = getitem_55.contiguous()
        getitem_55 = None
        layer_norm_17 = torch.nn.functional.layer_norm(
            x_101,
            (384,),
            l_self_modules_features_modules_5_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_101 = (
            l_self_modules_features_modules_5_modules_3_modules_norm1_parameters_weight_
        ) = (
            l_self_modules_features_modules_5_modules_3_modules_norm1_parameters_bias_
        ) = None
        _log_api_usage_once_14 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_14 = None
        x_102 = x_90 + layer_norm_17
        x_90 = layer_norm_17 = None
        input_63 = torch._C._nn.linear(
            x_102,
            l_self_modules_features_modules_5_modules_3_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_3_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_mlp_modules_0_parameters_bias_ = (None)
        input_64 = torch._C._nn.gelu(input_63, approximate="none")
        input_63 = None
        input_65 = torch.nn.functional.dropout(input_64, 0.0, False, False)
        input_64 = None
        input_66 = torch._C._nn.linear(
            input_65,
            l_self_modules_features_modules_5_modules_3_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_mlp_modules_3_parameters_bias_,
        )
        input_65 = l_self_modules_features_modules_5_modules_3_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_mlp_modules_3_parameters_bias_ = (None)
        input_67 = torch.nn.functional.dropout(input_66, 0.0, False, False)
        input_66 = None
        layer_norm_18 = torch.nn.functional.layer_norm(
            input_67,
            (384,),
            l_self_modules_features_modules_5_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_norm2_parameters_bias_,
            1e-05,
        )
        input_67 = (
            l_self_modules_features_modules_5_modules_3_modules_norm2_parameters_weight_
        ) = (
            l_self_modules_features_modules_5_modules_3_modules_norm2_parameters_bias_
        ) = None
        _log_api_usage_once_15 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_15 = None
        x_103 = x_102 + layer_norm_18
        x_102 = layer_norm_18 = None
        input_68 = torch._C._nn.linear(
            l_self_modules_features_modules_5_modules_4_modules_attn_buffers_relative_coords_table_,
            l_self_modules_features_modules_5_modules_4_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_4_modules_attn_buffers_relative_coords_table_ = l_self_modules_features_modules_5_modules_4_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = (None)
        input_69 = torch.nn.functional.relu(input_68, inplace=True)
        input_68 = None
        input_70 = torch._C._nn.linear(
            input_69,
            l_self_modules_features_modules_5_modules_4_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_,
            None,
        )
        input_69 = l_self_modules_features_modules_5_modules_4_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = (None)
        view_44 = input_70.view(-1, 12)
        input_70 = None
        relative_position_bias_32 = view_44[
            l_self_modules_features_modules_5_modules_4_modules_attn_buffers_relative_position_index_
        ]
        view_44 = l_self_modules_features_modules_5_modules_4_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_33 = relative_position_bias_32.view(64, 64, -1)
        relative_position_bias_32 = None
        permute_37 = relative_position_bias_33.permute(2, 0, 1)
        relative_position_bias_33 = None
        contiguous_16 = permute_37.contiguous()
        permute_37 = None
        relative_position_bias_34 = contiguous_16.unsqueeze(0)
        contiguous_16 = None
        sigmoid_8 = torch.sigmoid(relative_position_bias_34)
        relative_position_bias_34 = None
        relative_position_bias_35 = 16 * sigmoid_8
        sigmoid_8 = None
        x_104 = torch._C._nn.pad(x_103, (0, 0, 0, 2, 0, 2), "constant", None)
        x_105 = x_104.view(1, 2, 8, 2, 8, 384)
        x_104 = None
        permute_38 = x_105.permute(0, 1, 3, 2, 4, 5)
        x_105 = None
        x_106 = permute_38.reshape(4, 64, 384)
        permute_38 = None
        qkv_bias_8 = (
            l_self_modules_features_modules_5_modules_4_modules_attn_modules_qkv_parameters_bias_.clone()
        )
        l_self_modules_features_modules_5_modules_4_modules_attn_modules_qkv_parameters_bias_ = (
            None
        )
        getitem_57 = qkv_bias_8[slice(384, 768, None)]
        zero__8 = getitem_57.zero_()
        getitem_57 = zero__8 = None
        qkv_16 = torch._C._nn.linear(
            x_106,
            l_self_modules_features_modules_5_modules_4_modules_attn_modules_qkv_parameters_weight_,
            qkv_bias_8,
        )
        x_106 = l_self_modules_features_modules_5_modules_4_modules_attn_modules_qkv_parameters_weight_ = (qkv_bias_8) = (
            None
        )
        reshape_37 = qkv_16.reshape(4, 64, 3, 12, 32)
        qkv_16 = None
        qkv_17 = reshape_37.permute(2, 0, 3, 1, 4)
        reshape_37 = None
        q_8 = qkv_17[0]
        k_8 = qkv_17[1]
        v_8 = qkv_17[2]
        qkv_17 = None
        normalize_16 = torch.nn.functional.normalize(q_8, dim=-1)
        q_8 = None
        normalize_17 = torch.nn.functional.normalize(k_8, dim=-1)
        k_8 = None
        transpose_16 = normalize_17.transpose(-2, -1)
        normalize_17 = None
        attn_52 = normalize_16 @ transpose_16
        normalize_16 = transpose_16 = None
        clamp_8 = torch.clamp(
            l_self_modules_features_modules_5_modules_4_modules_attn_parameters_logit_scale_,
            max=4.605170185988092,
        )
        l_self_modules_features_modules_5_modules_4_modules_attn_parameters_logit_scale_ = (
            None
        )
        logit_scale_8 = clamp_8.exp()
        clamp_8 = None
        attn_53 = attn_52 * logit_scale_8
        attn_52 = logit_scale_8 = None
        attn_54 = attn_53 + relative_position_bias_35
        attn_53 = relative_position_bias_35 = None
        attn_55 = torch.nn.functional.softmax(attn_54, dim=-1)
        attn_54 = None
        attn_56 = torch.nn.functional.dropout(attn_55, p=0.0, training=False)
        attn_55 = None
        matmul_17 = attn_56.matmul(v_8)
        attn_56 = v_8 = None
        transpose_17 = matmul_17.transpose(1, 2)
        matmul_17 = None
        x_107 = transpose_17.reshape(4, 64, 384)
        transpose_17 = None
        x_108 = torch._C._nn.linear(
            x_107,
            l_self_modules_features_modules_5_modules_4_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_attn_modules_proj_parameters_bias_,
        )
        x_107 = l_self_modules_features_modules_5_modules_4_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_attn_modules_proj_parameters_bias_ = (None)
        x_109 = torch.nn.functional.dropout(x_108, p=0.0, training=False)
        x_108 = None
        x_110 = x_109.view(1, 2, 2, 8, 8, 384)
        x_109 = None
        permute_40 = x_110.permute(0, 1, 3, 2, 4, 5)
        x_110 = None
        x_111 = permute_40.reshape(1, 16, 16, 384)
        permute_40 = None
        getitem_61 = x_111[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_111 = None
        x_112 = getitem_61.contiguous()
        getitem_61 = None
        layer_norm_19 = torch.nn.functional.layer_norm(
            x_112,
            (384,),
            l_self_modules_features_modules_5_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_112 = (
            l_self_modules_features_modules_5_modules_4_modules_norm1_parameters_weight_
        ) = (
            l_self_modules_features_modules_5_modules_4_modules_norm1_parameters_bias_
        ) = None
        _log_api_usage_once_16 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_16 = None
        x_113 = x_103 + layer_norm_19
        x_103 = layer_norm_19 = None
        input_71 = torch._C._nn.linear(
            x_113,
            l_self_modules_features_modules_5_modules_4_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_4_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_mlp_modules_0_parameters_bias_ = (None)
        input_72 = torch._C._nn.gelu(input_71, approximate="none")
        input_71 = None
        input_73 = torch.nn.functional.dropout(input_72, 0.0, False, False)
        input_72 = None
        input_74 = torch._C._nn.linear(
            input_73,
            l_self_modules_features_modules_5_modules_4_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_mlp_modules_3_parameters_bias_,
        )
        input_73 = l_self_modules_features_modules_5_modules_4_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_mlp_modules_3_parameters_bias_ = (None)
        input_75 = torch.nn.functional.dropout(input_74, 0.0, False, False)
        input_74 = None
        layer_norm_20 = torch.nn.functional.layer_norm(
            input_75,
            (384,),
            l_self_modules_features_modules_5_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_norm2_parameters_bias_,
            1e-05,
        )
        input_75 = (
            l_self_modules_features_modules_5_modules_4_modules_norm2_parameters_weight_
        ) = (
            l_self_modules_features_modules_5_modules_4_modules_norm2_parameters_bias_
        ) = None
        _log_api_usage_once_17 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_17 = None
        x_114 = x_113 + layer_norm_20
        x_113 = layer_norm_20 = None
        input_76 = torch._C._nn.linear(
            l_self_modules_features_modules_5_modules_5_modules_attn_buffers_relative_coords_table_,
            l_self_modules_features_modules_5_modules_5_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_5_modules_attn_buffers_relative_coords_table_ = l_self_modules_features_modules_5_modules_5_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = (None)
        input_77 = torch.nn.functional.relu(input_76, inplace=True)
        input_76 = None
        input_78 = torch._C._nn.linear(
            input_77,
            l_self_modules_features_modules_5_modules_5_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_,
            None,
        )
        input_77 = l_self_modules_features_modules_5_modules_5_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = (None)
        view_48 = input_78.view(-1, 12)
        input_78 = None
        relative_position_bias_36 = view_48[
            l_self_modules_features_modules_5_modules_5_modules_attn_buffers_relative_position_index_
        ]
        view_48 = l_self_modules_features_modules_5_modules_5_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_37 = relative_position_bias_36.view(64, 64, -1)
        relative_position_bias_36 = None
        permute_41 = relative_position_bias_37.permute(2, 0, 1)
        relative_position_bias_37 = None
        contiguous_18 = permute_41.contiguous()
        permute_41 = None
        relative_position_bias_38 = contiguous_18.unsqueeze(0)
        contiguous_18 = None
        sigmoid_9 = torch.sigmoid(relative_position_bias_38)
        relative_position_bias_38 = None
        relative_position_bias_39 = 16 * sigmoid_9
        sigmoid_9 = None
        x_115 = torch._C._nn.pad(x_114, (0, 0, 0, 2, 0, 2), "constant", None)
        x_116 = torch.roll(x_115, shifts=(-4, -4), dims=(1, 2))
        x_115 = None
        x_117 = x_116.view(1, 2, 8, 2, 8, 384)
        x_116 = None
        permute_42 = x_117.permute(0, 1, 3, 2, 4, 5)
        x_117 = None
        x_118 = permute_42.reshape(4, 64, 384)
        permute_42 = None
        qkv_bias_9 = (
            l_self_modules_features_modules_5_modules_5_modules_attn_modules_qkv_parameters_bias_.clone()
        )
        l_self_modules_features_modules_5_modules_5_modules_attn_modules_qkv_parameters_bias_ = (
            None
        )
        getitem_63 = qkv_bias_9[slice(384, 768, None)]
        zero__9 = getitem_63.zero_()
        getitem_63 = zero__9 = None
        qkv_18 = torch._C._nn.linear(
            x_118,
            l_self_modules_features_modules_5_modules_5_modules_attn_modules_qkv_parameters_weight_,
            qkv_bias_9,
        )
        l_self_modules_features_modules_5_modules_5_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_9
        ) = None
        reshape_41 = qkv_18.reshape(4, 64, 3, 12, 32)
        qkv_18 = None
        qkv_19 = reshape_41.permute(2, 0, 3, 1, 4)
        reshape_41 = None
        q_9 = qkv_19[0]
        k_9 = qkv_19[1]
        v_9 = qkv_19[2]
        qkv_19 = None
        normalize_18 = torch.nn.functional.normalize(q_9, dim=-1)
        q_9 = None
        normalize_19 = torch.nn.functional.normalize(k_9, dim=-1)
        k_9 = None
        transpose_18 = normalize_19.transpose(-2, -1)
        normalize_19 = None
        attn_57 = normalize_18 @ transpose_18
        normalize_18 = transpose_18 = None
        clamp_9 = torch.clamp(
            l_self_modules_features_modules_5_modules_5_modules_attn_parameters_logit_scale_,
            max=4.605170185988092,
        )
        l_self_modules_features_modules_5_modules_5_modules_attn_parameters_logit_scale_ = (
            None
        )
        logit_scale_9 = clamp_9.exp()
        clamp_9 = None
        attn_58 = attn_57 * logit_scale_9
        attn_57 = logit_scale_9 = None
        attn_59 = attn_58 + relative_position_bias_39
        attn_58 = relative_position_bias_39 = None
        attn_mask_20 = x_118.new_zeros((16, 16))
        x_118 = None
        attn_mask_20[(slice(0, -8, None), slice(0, -8, None))] = 0
        setitem_36 = attn_mask_20
        setitem_36 = None
        attn_mask_20[(slice(0, -8, None), slice(-8, -4, None))] = 1
        setitem_37 = attn_mask_20
        setitem_37 = None
        attn_mask_20[(slice(0, -8, None), slice(-4, None, None))] = 2
        setitem_38 = attn_mask_20
        setitem_38 = None
        attn_mask_20[(slice(-8, -4, None), slice(0, -8, None))] = 3
        setitem_39 = attn_mask_20
        setitem_39 = None
        attn_mask_20[(slice(-8, -4, None), slice(-8, -4, None))] = 4
        setitem_40 = attn_mask_20
        setitem_40 = None
        attn_mask_20[(slice(-8, -4, None), slice(-4, None, None))] = 5
        setitem_41 = attn_mask_20
        setitem_41 = None
        attn_mask_20[(slice(-4, None, None), slice(0, -8, None))] = 6
        setitem_42 = attn_mask_20
        setitem_42 = None
        attn_mask_20[(slice(-4, None, None), slice(-8, -4, None))] = 7
        setitem_43 = attn_mask_20
        setitem_43 = None
        attn_mask_20[(slice(-4, None, None), slice(-4, None, None))] = 8
        setitem_44 = attn_mask_20
        setitem_44 = None
        attn_mask_21 = attn_mask_20.view(2, 8, 2, 8)
        attn_mask_20 = None
        permute_44 = attn_mask_21.permute(0, 2, 1, 3)
        attn_mask_21 = None
        attn_mask_22 = permute_44.reshape(4, 64)
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
        attn_60 = attn_59.view(1, 4, 12, 64, 64)
        attn_59 = None
        unsqueeze_28 = attn_mask_24.unsqueeze(1)
        attn_mask_24 = None
        unsqueeze_29 = unsqueeze_28.unsqueeze(0)
        unsqueeze_28 = None
        attn_61 = attn_60 + unsqueeze_29
        attn_60 = unsqueeze_29 = None
        attn_62 = attn_61.view(-1, 12, 64, 64)
        attn_61 = None
        attn_63 = torch.nn.functional.softmax(attn_62, dim=-1)
        attn_62 = None
        attn_64 = torch.nn.functional.dropout(attn_63, p=0.0, training=False)
        attn_63 = None
        matmul_19 = attn_64.matmul(v_9)
        attn_64 = v_9 = None
        transpose_19 = matmul_19.transpose(1, 2)
        matmul_19 = None
        x_119 = transpose_19.reshape(4, 64, 384)
        transpose_19 = None
        x_120 = torch._C._nn.linear(
            x_119,
            l_self_modules_features_modules_5_modules_5_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_attn_modules_proj_parameters_bias_,
        )
        x_119 = l_self_modules_features_modules_5_modules_5_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_attn_modules_proj_parameters_bias_ = (None)
        x_121 = torch.nn.functional.dropout(x_120, p=0.0, training=False)
        x_120 = None
        x_122 = x_121.view(1, 2, 2, 8, 8, 384)
        x_121 = None
        permute_45 = x_122.permute(0, 1, 3, 2, 4, 5)
        x_122 = None
        x_123 = permute_45.reshape(1, 16, 16, 384)
        permute_45 = None
        x_124 = torch.roll(x_123, shifts=(4, 4), dims=(1, 2))
        x_123 = None
        getitem_67 = x_124[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_124 = None
        x_125 = getitem_67.contiguous()
        getitem_67 = None
        layer_norm_21 = torch.nn.functional.layer_norm(
            x_125,
            (384,),
            l_self_modules_features_modules_5_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_125 = (
            l_self_modules_features_modules_5_modules_5_modules_norm1_parameters_weight_
        ) = (
            l_self_modules_features_modules_5_modules_5_modules_norm1_parameters_bias_
        ) = None
        _log_api_usage_once_18 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_18 = None
        x_126 = x_114 + layer_norm_21
        x_114 = layer_norm_21 = None
        input_79 = torch._C._nn.linear(
            x_126,
            l_self_modules_features_modules_5_modules_5_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_5_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_mlp_modules_0_parameters_bias_ = (None)
        input_80 = torch._C._nn.gelu(input_79, approximate="none")
        input_79 = None
        input_81 = torch.nn.functional.dropout(input_80, 0.0, False, False)
        input_80 = None
        input_82 = torch._C._nn.linear(
            input_81,
            l_self_modules_features_modules_5_modules_5_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_mlp_modules_3_parameters_bias_,
        )
        input_81 = l_self_modules_features_modules_5_modules_5_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_mlp_modules_3_parameters_bias_ = (None)
        input_83 = torch.nn.functional.dropout(input_82, 0.0, False, False)
        input_82 = None
        layer_norm_22 = torch.nn.functional.layer_norm(
            input_83,
            (384,),
            l_self_modules_features_modules_5_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_norm2_parameters_bias_,
            1e-05,
        )
        input_83 = (
            l_self_modules_features_modules_5_modules_5_modules_norm2_parameters_weight_
        ) = (
            l_self_modules_features_modules_5_modules_5_modules_norm2_parameters_bias_
        ) = None
        _log_api_usage_once_19 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_19 = None
        x_127 = x_126 + layer_norm_22
        x_126 = layer_norm_22 = None
        x_128 = torch._C._nn.pad(x_127, (0, 0, 0, 0, 0, 0), "constant", None)
        x_127 = None
        x0_2 = x_128[
            (Ellipsis, slice(0, None, 2), slice(0, None, 2), slice(None, None, None))
        ]
        x1_2 = x_128[
            (Ellipsis, slice(1, None, 2), slice(0, None, 2), slice(None, None, None))
        ]
        x2_2 = x_128[
            (Ellipsis, slice(0, None, 2), slice(1, None, 2), slice(None, None, None))
        ]
        x3_2 = x_128[
            (Ellipsis, slice(1, None, 2), slice(1, None, 2), slice(None, None, None))
        ]
        x_128 = None
        x_129 = torch.cat([x0_2, x1_2, x2_2, x3_2], -1)
        x0_2 = x1_2 = x2_2 = x3_2 = None
        x_130 = torch._C._nn.linear(
            x_129,
            l_self_modules_features_modules_6_modules_reduction_parameters_weight_,
            None,
        )
        x_129 = (
            l_self_modules_features_modules_6_modules_reduction_parameters_weight_
        ) = None
        x_131 = torch.nn.functional.layer_norm(
            x_130,
            (768,),
            l_self_modules_features_modules_6_modules_norm_parameters_weight_,
            l_self_modules_features_modules_6_modules_norm_parameters_bias_,
            1e-05,
        )
        x_130 = (
            l_self_modules_features_modules_6_modules_norm_parameters_weight_
        ) = l_self_modules_features_modules_6_modules_norm_parameters_bias_ = None
        input_84 = torch._C._nn.linear(
            l_self_modules_features_modules_7_modules_0_modules_attn_buffers_relative_coords_table_,
            l_self_modules_features_modules_7_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_7_modules_0_modules_attn_buffers_relative_coords_table_ = l_self_modules_features_modules_7_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = (None)
        input_85 = torch.nn.functional.relu(input_84, inplace=True)
        input_84 = None
        input_86 = torch._C._nn.linear(
            input_85,
            l_self_modules_features_modules_7_modules_0_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_,
            None,
        )
        input_85 = l_self_modules_features_modules_7_modules_0_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = (None)
        view_55 = input_86.view(-1, 24)
        input_86 = None
        relative_position_bias_40 = view_55[
            l_self_modules_features_modules_7_modules_0_modules_attn_buffers_relative_position_index_
        ]
        view_55 = l_self_modules_features_modules_7_modules_0_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_41 = relative_position_bias_40.view(64, 64, -1)
        relative_position_bias_40 = None
        permute_46 = relative_position_bias_41.permute(2, 0, 1)
        relative_position_bias_41 = None
        contiguous_20 = permute_46.contiguous()
        permute_46 = None
        relative_position_bias_42 = contiguous_20.unsqueeze(0)
        contiguous_20 = None
        sigmoid_10 = torch.sigmoid(relative_position_bias_42)
        relative_position_bias_42 = None
        relative_position_bias_43 = 16 * sigmoid_10
        sigmoid_10 = None
        x_132 = torch._C._nn.pad(x_131, (0, 0, 0, 1, 0, 1), "constant", None)
        x_133 = x_132.view(1, 1, 8, 1, 8, 768)
        x_132 = None
        permute_47 = x_133.permute(0, 1, 3, 2, 4, 5)
        x_133 = None
        x_134 = permute_47.reshape(1, 64, 768)
        permute_47 = None
        qkv_bias_10 = (
            l_self_modules_features_modules_7_modules_0_modules_attn_modules_qkv_parameters_bias_.clone()
        )
        l_self_modules_features_modules_7_modules_0_modules_attn_modules_qkv_parameters_bias_ = (
            None
        )
        getitem_73 = qkv_bias_10[slice(768, 1536, None)]
        zero__10 = getitem_73.zero_()
        getitem_73 = zero__10 = None
        qkv_20 = torch._C._nn.linear(
            x_134,
            l_self_modules_features_modules_7_modules_0_modules_attn_modules_qkv_parameters_weight_,
            qkv_bias_10,
        )
        x_134 = l_self_modules_features_modules_7_modules_0_modules_attn_modules_qkv_parameters_weight_ = (qkv_bias_10) = (
            None
        )
        reshape_46 = qkv_20.reshape(1, 64, 3, 24, 32)
        qkv_20 = None
        qkv_21 = reshape_46.permute(2, 0, 3, 1, 4)
        reshape_46 = None
        q_10 = qkv_21[0]
        k_10 = qkv_21[1]
        v_10 = qkv_21[2]
        qkv_21 = None
        normalize_20 = torch.nn.functional.normalize(q_10, dim=-1)
        q_10 = None
        normalize_21 = torch.nn.functional.normalize(k_10, dim=-1)
        k_10 = None
        transpose_20 = normalize_21.transpose(-2, -1)
        normalize_21 = None
        attn_65 = normalize_20 @ transpose_20
        normalize_20 = transpose_20 = None
        clamp_10 = torch.clamp(
            l_self_modules_features_modules_7_modules_0_modules_attn_parameters_logit_scale_,
            max=4.605170185988092,
        )
        l_self_modules_features_modules_7_modules_0_modules_attn_parameters_logit_scale_ = (
            None
        )
        logit_scale_10 = clamp_10.exp()
        clamp_10 = None
        attn_66 = attn_65 * logit_scale_10
        attn_65 = logit_scale_10 = None
        attn_67 = attn_66 + relative_position_bias_43
        attn_66 = relative_position_bias_43 = None
        attn_68 = torch.nn.functional.softmax(attn_67, dim=-1)
        attn_67 = None
        attn_69 = torch.nn.functional.dropout(attn_68, p=0.0, training=False)
        attn_68 = None
        matmul_21 = attn_69.matmul(v_10)
        attn_69 = v_10 = None
        transpose_21 = matmul_21.transpose(1, 2)
        matmul_21 = None
        x_135 = transpose_21.reshape(1, 64, 768)
        transpose_21 = None
        x_136 = torch._C._nn.linear(
            x_135,
            l_self_modules_features_modules_7_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_135 = l_self_modules_features_modules_7_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_137 = torch.nn.functional.dropout(x_136, p=0.0, training=False)
        x_136 = None
        x_138 = x_137.view(1, 1, 1, 8, 8, 768)
        x_137 = None
        permute_49 = x_138.permute(0, 1, 3, 2, 4, 5)
        x_138 = None
        x_139 = permute_49.reshape(1, 8, 8, 768)
        permute_49 = None
        getitem_77 = x_139[
            (
                slice(None, None, None),
                slice(None, 7, None),
                slice(None, 7, None),
                slice(None, None, None),
            )
        ]
        x_139 = None
        x_140 = getitem_77.contiguous()
        getitem_77 = None
        layer_norm_24 = torch.nn.functional.layer_norm(
            x_140,
            (768,),
            l_self_modules_features_modules_7_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_140 = (
            l_self_modules_features_modules_7_modules_0_modules_norm1_parameters_weight_
        ) = (
            l_self_modules_features_modules_7_modules_0_modules_norm1_parameters_bias_
        ) = None
        _log_api_usage_once_20 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_20 = None
        x_141 = x_131 + layer_norm_24
        x_131 = layer_norm_24 = None
        input_87 = torch._C._nn.linear(
            x_141,
            l_self_modules_features_modules_7_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_7_modules_0_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_mlp_modules_0_parameters_bias_ = (None)
        input_88 = torch._C._nn.gelu(input_87, approximate="none")
        input_87 = None
        input_89 = torch.nn.functional.dropout(input_88, 0.0, False, False)
        input_88 = None
        input_90 = torch._C._nn.linear(
            input_89,
            l_self_modules_features_modules_7_modules_0_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_mlp_modules_3_parameters_bias_,
        )
        input_89 = l_self_modules_features_modules_7_modules_0_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_mlp_modules_3_parameters_bias_ = (None)
        input_91 = torch.nn.functional.dropout(input_90, 0.0, False, False)
        input_90 = None
        layer_norm_25 = torch.nn.functional.layer_norm(
            input_91,
            (768,),
            l_self_modules_features_modules_7_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        input_91 = (
            l_self_modules_features_modules_7_modules_0_modules_norm2_parameters_weight_
        ) = (
            l_self_modules_features_modules_7_modules_0_modules_norm2_parameters_bias_
        ) = None
        _log_api_usage_once_21 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_21 = None
        x_142 = x_141 + layer_norm_25
        x_141 = layer_norm_25 = None
        input_92 = torch._C._nn.linear(
            l_self_modules_features_modules_7_modules_1_modules_attn_buffers_relative_coords_table_,
            l_self_modules_features_modules_7_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_7_modules_1_modules_attn_buffers_relative_coords_table_ = l_self_modules_features_modules_7_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = (None)
        input_93 = torch.nn.functional.relu(input_92, inplace=True)
        input_92 = None
        input_94 = torch._C._nn.linear(
            input_93,
            l_self_modules_features_modules_7_modules_1_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_,
            None,
        )
        input_93 = l_self_modules_features_modules_7_modules_1_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = (None)
        view_59 = input_94.view(-1, 24)
        input_94 = None
        relative_position_bias_44 = view_59[
            l_self_modules_features_modules_7_modules_1_modules_attn_buffers_relative_position_index_
        ]
        view_59 = l_self_modules_features_modules_7_modules_1_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_45 = relative_position_bias_44.view(64, 64, -1)
        relative_position_bias_44 = None
        permute_50 = relative_position_bias_45.permute(2, 0, 1)
        relative_position_bias_45 = None
        contiguous_22 = permute_50.contiguous()
        permute_50 = None
        relative_position_bias_46 = contiguous_22.unsqueeze(0)
        contiguous_22 = None
        sigmoid_11 = torch.sigmoid(relative_position_bias_46)
        relative_position_bias_46 = None
        relative_position_bias_47 = 16 * sigmoid_11
        sigmoid_11 = None
        x_143 = torch._C._nn.pad(x_142, (0, 0, 0, 1, 0, 1), "constant", None)
        x_144 = x_143.view(1, 1, 8, 1, 8, 768)
        x_143 = None
        permute_51 = x_144.permute(0, 1, 3, 2, 4, 5)
        x_144 = None
        x_145 = permute_51.reshape(1, 64, 768)
        permute_51 = None
        qkv_bias_11 = (
            l_self_modules_features_modules_7_modules_1_modules_attn_modules_qkv_parameters_bias_.clone()
        )
        l_self_modules_features_modules_7_modules_1_modules_attn_modules_qkv_parameters_bias_ = (
            None
        )
        getitem_79 = qkv_bias_11[slice(768, 1536, None)]
        zero__11 = getitem_79.zero_()
        getitem_79 = zero__11 = None
        qkv_22 = torch._C._nn.linear(
            x_145,
            l_self_modules_features_modules_7_modules_1_modules_attn_modules_qkv_parameters_weight_,
            qkv_bias_11,
        )
        x_145 = l_self_modules_features_modules_7_modules_1_modules_attn_modules_qkv_parameters_weight_ = (qkv_bias_11) = (
            None
        )
        reshape_50 = qkv_22.reshape(1, 64, 3, 24, 32)
        qkv_22 = None
        qkv_23 = reshape_50.permute(2, 0, 3, 1, 4)
        reshape_50 = None
        q_11 = qkv_23[0]
        k_11 = qkv_23[1]
        v_11 = qkv_23[2]
        qkv_23 = None
        normalize_22 = torch.nn.functional.normalize(q_11, dim=-1)
        q_11 = None
        normalize_23 = torch.nn.functional.normalize(k_11, dim=-1)
        k_11 = None
        transpose_22 = normalize_23.transpose(-2, -1)
        normalize_23 = None
        attn_70 = normalize_22 @ transpose_22
        normalize_22 = transpose_22 = None
        clamp_11 = torch.clamp(
            l_self_modules_features_modules_7_modules_1_modules_attn_parameters_logit_scale_,
            max=4.605170185988092,
        )
        l_self_modules_features_modules_7_modules_1_modules_attn_parameters_logit_scale_ = (
            None
        )
        logit_scale_11 = clamp_11.exp()
        clamp_11 = None
        attn_71 = attn_70 * logit_scale_11
        attn_70 = logit_scale_11 = None
        attn_72 = attn_71 + relative_position_bias_47
        attn_71 = relative_position_bias_47 = None
        attn_73 = torch.nn.functional.softmax(attn_72, dim=-1)
        attn_72 = None
        attn_74 = torch.nn.functional.dropout(attn_73, p=0.0, training=False)
        attn_73 = None
        matmul_23 = attn_74.matmul(v_11)
        attn_74 = v_11 = None
        transpose_23 = matmul_23.transpose(1, 2)
        matmul_23 = None
        x_146 = transpose_23.reshape(1, 64, 768)
        transpose_23 = None
        x_147 = torch._C._nn.linear(
            x_146,
            l_self_modules_features_modules_7_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_146 = l_self_modules_features_modules_7_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_148 = torch.nn.functional.dropout(x_147, p=0.0, training=False)
        x_147 = None
        x_149 = x_148.view(1, 1, 1, 8, 8, 768)
        x_148 = None
        permute_53 = x_149.permute(0, 1, 3, 2, 4, 5)
        x_149 = None
        x_150 = permute_53.reshape(1, 8, 8, 768)
        permute_53 = None
        getitem_83 = x_150[
            (
                slice(None, None, None),
                slice(None, 7, None),
                slice(None, 7, None),
                slice(None, None, None),
            )
        ]
        x_150 = None
        x_151 = getitem_83.contiguous()
        getitem_83 = None
        layer_norm_26 = torch.nn.functional.layer_norm(
            x_151,
            (768,),
            l_self_modules_features_modules_7_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_151 = (
            l_self_modules_features_modules_7_modules_1_modules_norm1_parameters_weight_
        ) = (
            l_self_modules_features_modules_7_modules_1_modules_norm1_parameters_bias_
        ) = None
        _log_api_usage_once_22 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_22 = None
        x_152 = x_142 + layer_norm_26
        x_142 = layer_norm_26 = None
        input_95 = torch._C._nn.linear(
            x_152,
            l_self_modules_features_modules_7_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_7_modules_1_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_mlp_modules_0_parameters_bias_ = (None)
        input_96 = torch._C._nn.gelu(input_95, approximate="none")
        input_95 = None
        input_97 = torch.nn.functional.dropout(input_96, 0.0, False, False)
        input_96 = None
        input_98 = torch._C._nn.linear(
            input_97,
            l_self_modules_features_modules_7_modules_1_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_mlp_modules_3_parameters_bias_,
        )
        input_97 = l_self_modules_features_modules_7_modules_1_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_mlp_modules_3_parameters_bias_ = (None)
        input_99 = torch.nn.functional.dropout(input_98, 0.0, False, False)
        input_98 = None
        layer_norm_27 = torch.nn.functional.layer_norm(
            input_99,
            (768,),
            l_self_modules_features_modules_7_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        input_99 = (
            l_self_modules_features_modules_7_modules_1_modules_norm2_parameters_weight_
        ) = (
            l_self_modules_features_modules_7_modules_1_modules_norm2_parameters_bias_
        ) = None
        _log_api_usage_once_23 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_23 = None
        x_153 = x_152 + layer_norm_27
        x_152 = layer_norm_27 = None
        x_154 = torch.nn.functional.layer_norm(
            x_153,
            (768,),
            l_self_modules_norm_parameters_weight_,
            l_self_modules_norm_parameters_bias_,
            1e-05,
        )
        x_153 = (
            l_self_modules_norm_parameters_weight_
        ) = l_self_modules_norm_parameters_bias_ = None
        x_155 = torch.permute(x_154, [0, 3, 1, 2])
        x_154 = None
        x_156 = torch.nn.functional.adaptive_avg_pool2d(x_155, 1)
        x_155 = None
        x_157 = x_156.flatten(1, -1)
        x_156 = None
        x_158 = torch._C._nn.linear(
            x_157,
            l_self_modules_head_parameters_weight_,
            l_self_modules_head_parameters_bias_,
        )
        x_157 = (
            l_self_modules_head_parameters_weight_
        ) = l_self_modules_head_parameters_bias_ = None
        return (x_158,)
