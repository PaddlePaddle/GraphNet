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
        L_self_modules_features_modules_5_modules_6_modules_attn_buffers_relative_coords_table_: torch.Tensor,
        L_self_modules_features_modules_5_modules_6_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_5_modules_6_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_attn_buffers_relative_coords_table_: torch.Tensor,
        L_self_modules_features_modules_5_modules_7_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_5_modules_7_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_attn_buffers_relative_coords_table_: torch.Tensor,
        L_self_modules_features_modules_5_modules_8_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_5_modules_8_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_attn_buffers_relative_coords_table_: torch.Tensor,
        L_self_modules_features_modules_5_modules_9_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_5_modules_9_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_attn_buffers_relative_coords_table_: torch.Tensor,
        L_self_modules_features_modules_5_modules_10_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_5_modules_10_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_attn_buffers_relative_coords_table_: torch.Tensor,
        L_self_modules_features_modules_5_modules_11_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_5_modules_11_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_attn_buffers_relative_coords_table_: torch.Tensor,
        L_self_modules_features_modules_5_modules_12_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_5_modules_12_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_attn_buffers_relative_coords_table_: torch.Tensor,
        L_self_modules_features_modules_5_modules_13_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_5_modules_13_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_attn_buffers_relative_coords_table_: torch.Tensor,
        L_self_modules_features_modules_5_modules_14_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_5_modules_14_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_attn_buffers_relative_coords_table_: torch.Tensor,
        L_self_modules_features_modules_5_modules_15_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_5_modules_15_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_attn_buffers_relative_coords_table_: torch.Tensor,
        L_self_modules_features_modules_5_modules_16_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_5_modules_16_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_attn_buffers_relative_coords_table_: torch.Tensor,
        L_self_modules_features_modules_5_modules_17_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_features_modules_5_modules_17_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_attn_parameters_logit_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_features_modules_5_modules_6_modules_attn_buffers_relative_coords_table_ = L_self_modules_features_modules_5_modules_6_modules_attn_buffers_relative_coords_table_
        l_self_modules_features_modules_5_modules_6_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_6_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_6_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_6_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_6_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_6_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_6_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_5_modules_6_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_5_modules_6_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_5_modules_6_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_5_modules_6_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_5_modules_6_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_5_modules_6_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_5_modules_6_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_5_modules_6_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_5_modules_6_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_5_modules_6_modules_attn_parameters_logit_scale_ = L_self_modules_features_modules_5_modules_6_modules_attn_parameters_logit_scale_
        l_self_modules_features_modules_5_modules_6_modules_norm1_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_6_modules_norm1_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_6_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_6_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_6_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_6_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_6_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_6_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_6_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_6_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_6_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_6_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_6_modules_norm2_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_6_modules_norm2_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_6_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_6_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_7_modules_attn_buffers_relative_coords_table_ = L_self_modules_features_modules_5_modules_7_modules_attn_buffers_relative_coords_table_
        l_self_modules_features_modules_5_modules_7_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_7_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_7_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_7_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_7_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_7_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_7_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_5_modules_7_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_5_modules_7_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_5_modules_7_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_5_modules_7_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_5_modules_7_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_5_modules_7_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_5_modules_7_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_5_modules_7_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_5_modules_7_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_5_modules_7_modules_attn_parameters_logit_scale_ = L_self_modules_features_modules_5_modules_7_modules_attn_parameters_logit_scale_
        l_self_modules_features_modules_5_modules_7_modules_norm1_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_7_modules_norm1_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_7_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_7_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_7_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_7_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_7_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_7_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_7_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_7_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_7_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_7_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_7_modules_norm2_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_7_modules_norm2_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_7_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_7_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_8_modules_attn_buffers_relative_coords_table_ = L_self_modules_features_modules_5_modules_8_modules_attn_buffers_relative_coords_table_
        l_self_modules_features_modules_5_modules_8_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_8_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_8_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_8_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_8_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_8_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_8_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_5_modules_8_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_5_modules_8_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_5_modules_8_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_5_modules_8_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_5_modules_8_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_5_modules_8_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_5_modules_8_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_5_modules_8_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_5_modules_8_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_5_modules_8_modules_attn_parameters_logit_scale_ = L_self_modules_features_modules_5_modules_8_modules_attn_parameters_logit_scale_
        l_self_modules_features_modules_5_modules_8_modules_norm1_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_8_modules_norm1_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_8_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_8_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_8_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_8_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_8_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_8_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_8_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_8_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_8_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_8_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_8_modules_norm2_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_8_modules_norm2_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_8_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_8_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_9_modules_attn_buffers_relative_coords_table_ = L_self_modules_features_modules_5_modules_9_modules_attn_buffers_relative_coords_table_
        l_self_modules_features_modules_5_modules_9_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_9_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_9_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_9_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_9_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_9_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_9_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_5_modules_9_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_5_modules_9_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_5_modules_9_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_5_modules_9_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_5_modules_9_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_5_modules_9_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_5_modules_9_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_5_modules_9_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_5_modules_9_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_5_modules_9_modules_attn_parameters_logit_scale_ = L_self_modules_features_modules_5_modules_9_modules_attn_parameters_logit_scale_
        l_self_modules_features_modules_5_modules_9_modules_norm1_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_9_modules_norm1_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_9_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_9_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_9_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_9_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_9_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_9_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_9_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_9_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_9_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_9_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_9_modules_norm2_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_9_modules_norm2_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_9_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_9_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_10_modules_attn_buffers_relative_coords_table_ = L_self_modules_features_modules_5_modules_10_modules_attn_buffers_relative_coords_table_
        l_self_modules_features_modules_5_modules_10_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_10_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_10_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_10_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_10_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_10_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_10_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_5_modules_10_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_5_modules_10_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_5_modules_10_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_5_modules_10_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_5_modules_10_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_5_modules_10_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_5_modules_10_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_5_modules_10_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_5_modules_10_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_5_modules_10_modules_attn_parameters_logit_scale_ = L_self_modules_features_modules_5_modules_10_modules_attn_parameters_logit_scale_
        l_self_modules_features_modules_5_modules_10_modules_norm1_parameters_weight_ = L_self_modules_features_modules_5_modules_10_modules_norm1_parameters_weight_
        l_self_modules_features_modules_5_modules_10_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_10_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_10_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_10_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_10_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_10_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_10_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_10_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_10_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_10_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_10_modules_norm2_parameters_weight_ = L_self_modules_features_modules_5_modules_10_modules_norm2_parameters_weight_
        l_self_modules_features_modules_5_modules_10_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_10_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_11_modules_attn_buffers_relative_coords_table_ = L_self_modules_features_modules_5_modules_11_modules_attn_buffers_relative_coords_table_
        l_self_modules_features_modules_5_modules_11_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_11_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_11_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_11_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_11_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_11_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_11_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_5_modules_11_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_5_modules_11_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_5_modules_11_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_5_modules_11_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_5_modules_11_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_5_modules_11_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_5_modules_11_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_5_modules_11_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_5_modules_11_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_5_modules_11_modules_attn_parameters_logit_scale_ = L_self_modules_features_modules_5_modules_11_modules_attn_parameters_logit_scale_
        l_self_modules_features_modules_5_modules_11_modules_norm1_parameters_weight_ = L_self_modules_features_modules_5_modules_11_modules_norm1_parameters_weight_
        l_self_modules_features_modules_5_modules_11_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_11_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_11_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_11_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_11_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_11_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_11_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_11_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_11_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_11_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_11_modules_norm2_parameters_weight_ = L_self_modules_features_modules_5_modules_11_modules_norm2_parameters_weight_
        l_self_modules_features_modules_5_modules_11_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_11_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_12_modules_attn_buffers_relative_coords_table_ = L_self_modules_features_modules_5_modules_12_modules_attn_buffers_relative_coords_table_
        l_self_modules_features_modules_5_modules_12_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_12_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_12_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_12_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_12_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_12_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_12_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_5_modules_12_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_5_modules_12_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_5_modules_12_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_5_modules_12_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_5_modules_12_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_5_modules_12_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_5_modules_12_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_5_modules_12_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_5_modules_12_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_5_modules_12_modules_attn_parameters_logit_scale_ = L_self_modules_features_modules_5_modules_12_modules_attn_parameters_logit_scale_
        l_self_modules_features_modules_5_modules_12_modules_norm1_parameters_weight_ = L_self_modules_features_modules_5_modules_12_modules_norm1_parameters_weight_
        l_self_modules_features_modules_5_modules_12_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_12_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_12_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_12_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_12_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_12_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_12_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_12_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_12_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_12_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_12_modules_norm2_parameters_weight_ = L_self_modules_features_modules_5_modules_12_modules_norm2_parameters_weight_
        l_self_modules_features_modules_5_modules_12_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_12_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_13_modules_attn_buffers_relative_coords_table_ = L_self_modules_features_modules_5_modules_13_modules_attn_buffers_relative_coords_table_
        l_self_modules_features_modules_5_modules_13_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_13_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_13_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_13_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_13_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_13_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_13_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_5_modules_13_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_5_modules_13_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_5_modules_13_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_5_modules_13_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_5_modules_13_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_5_modules_13_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_5_modules_13_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_5_modules_13_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_5_modules_13_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_5_modules_13_modules_attn_parameters_logit_scale_ = L_self_modules_features_modules_5_modules_13_modules_attn_parameters_logit_scale_
        l_self_modules_features_modules_5_modules_13_modules_norm1_parameters_weight_ = L_self_modules_features_modules_5_modules_13_modules_norm1_parameters_weight_
        l_self_modules_features_modules_5_modules_13_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_13_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_13_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_13_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_13_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_13_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_13_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_13_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_13_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_13_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_13_modules_norm2_parameters_weight_ = L_self_modules_features_modules_5_modules_13_modules_norm2_parameters_weight_
        l_self_modules_features_modules_5_modules_13_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_13_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_14_modules_attn_buffers_relative_coords_table_ = L_self_modules_features_modules_5_modules_14_modules_attn_buffers_relative_coords_table_
        l_self_modules_features_modules_5_modules_14_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_14_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_14_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_14_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_14_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_14_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_14_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_5_modules_14_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_5_modules_14_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_5_modules_14_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_5_modules_14_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_5_modules_14_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_5_modules_14_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_5_modules_14_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_5_modules_14_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_5_modules_14_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_5_modules_14_modules_attn_parameters_logit_scale_ = L_self_modules_features_modules_5_modules_14_modules_attn_parameters_logit_scale_
        l_self_modules_features_modules_5_modules_14_modules_norm1_parameters_weight_ = L_self_modules_features_modules_5_modules_14_modules_norm1_parameters_weight_
        l_self_modules_features_modules_5_modules_14_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_14_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_14_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_14_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_14_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_14_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_14_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_14_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_14_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_14_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_14_modules_norm2_parameters_weight_ = L_self_modules_features_modules_5_modules_14_modules_norm2_parameters_weight_
        l_self_modules_features_modules_5_modules_14_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_14_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_15_modules_attn_buffers_relative_coords_table_ = L_self_modules_features_modules_5_modules_15_modules_attn_buffers_relative_coords_table_
        l_self_modules_features_modules_5_modules_15_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_15_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_15_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_15_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_15_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_15_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_15_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_5_modules_15_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_5_modules_15_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_5_modules_15_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_5_modules_15_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_5_modules_15_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_5_modules_15_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_5_modules_15_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_5_modules_15_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_5_modules_15_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_5_modules_15_modules_attn_parameters_logit_scale_ = L_self_modules_features_modules_5_modules_15_modules_attn_parameters_logit_scale_
        l_self_modules_features_modules_5_modules_15_modules_norm1_parameters_weight_ = L_self_modules_features_modules_5_modules_15_modules_norm1_parameters_weight_
        l_self_modules_features_modules_5_modules_15_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_15_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_15_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_15_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_15_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_15_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_15_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_15_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_15_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_15_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_15_modules_norm2_parameters_weight_ = L_self_modules_features_modules_5_modules_15_modules_norm2_parameters_weight_
        l_self_modules_features_modules_5_modules_15_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_15_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_16_modules_attn_buffers_relative_coords_table_ = L_self_modules_features_modules_5_modules_16_modules_attn_buffers_relative_coords_table_
        l_self_modules_features_modules_5_modules_16_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_16_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_16_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_16_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_16_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_16_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_16_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_5_modules_16_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_5_modules_16_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_5_modules_16_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_5_modules_16_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_5_modules_16_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_5_modules_16_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_5_modules_16_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_5_modules_16_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_5_modules_16_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_5_modules_16_modules_attn_parameters_logit_scale_ = L_self_modules_features_modules_5_modules_16_modules_attn_parameters_logit_scale_
        l_self_modules_features_modules_5_modules_16_modules_norm1_parameters_weight_ = L_self_modules_features_modules_5_modules_16_modules_norm1_parameters_weight_
        l_self_modules_features_modules_5_modules_16_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_16_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_16_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_16_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_16_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_16_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_16_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_16_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_16_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_16_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_16_modules_norm2_parameters_weight_ = L_self_modules_features_modules_5_modules_16_modules_norm2_parameters_weight_
        l_self_modules_features_modules_5_modules_16_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_16_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_17_modules_attn_buffers_relative_coords_table_ = L_self_modules_features_modules_5_modules_17_modules_attn_buffers_relative_coords_table_
        l_self_modules_features_modules_5_modules_17_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_17_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_17_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_17_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_17_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_17_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_17_modules_attn_buffers_relative_position_index_ = L_self_modules_features_modules_5_modules_17_modules_attn_buffers_relative_position_index_
        l_self_modules_features_modules_5_modules_17_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_features_modules_5_modules_17_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_features_modules_5_modules_17_modules_attn_modules_proj_parameters_weight_ = L_self_modules_features_modules_5_modules_17_modules_attn_modules_proj_parameters_weight_
        l_self_modules_features_modules_5_modules_17_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_features_modules_5_modules_17_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_features_modules_5_modules_17_modules_attn_modules_proj_parameters_bias_ = L_self_modules_features_modules_5_modules_17_modules_attn_modules_proj_parameters_bias_
        l_self_modules_features_modules_5_modules_17_modules_attn_parameters_logit_scale_ = L_self_modules_features_modules_5_modules_17_modules_attn_parameters_logit_scale_
        l_self_modules_features_modules_5_modules_17_modules_norm1_parameters_weight_ = L_self_modules_features_modules_5_modules_17_modules_norm1_parameters_weight_
        l_self_modules_features_modules_5_modules_17_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_17_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_17_modules_mlp_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_17_modules_mlp_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_17_modules_mlp_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_17_modules_mlp_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_17_modules_mlp_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_17_modules_mlp_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_17_modules_mlp_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_17_modules_mlp_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_17_modules_norm2_parameters_weight_ = L_self_modules_features_modules_5_modules_17_modules_norm2_parameters_weight_
        l_self_modules_features_modules_5_modules_17_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_17_modules_norm2_parameters_bias_
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
        input_84 = torch._C._nn.linear(
            l_self_modules_features_modules_5_modules_6_modules_attn_buffers_relative_coords_table_,
            l_self_modules_features_modules_5_modules_6_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_6_modules_attn_buffers_relative_coords_table_ = l_self_modules_features_modules_5_modules_6_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = (None)
        input_85 = torch.nn.functional.relu(input_84, inplace=True)
        input_84 = None
        input_86 = torch._C._nn.linear(
            input_85,
            l_self_modules_features_modules_5_modules_6_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_,
            None,
        )
        input_85 = l_self_modules_features_modules_5_modules_6_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = (None)
        view_55 = input_86.view(-1, 12)
        input_86 = None
        relative_position_bias_40 = view_55[
            l_self_modules_features_modules_5_modules_6_modules_attn_buffers_relative_position_index_
        ]
        view_55 = l_self_modules_features_modules_5_modules_6_modules_attn_buffers_relative_position_index_ = (None)
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
        x_128 = torch._C._nn.pad(x_127, (0, 0, 0, 2, 0, 2), "constant", None)
        x_129 = x_128.view(1, 2, 8, 2, 8, 384)
        x_128 = None
        permute_47 = x_129.permute(0, 1, 3, 2, 4, 5)
        x_129 = None
        x_130 = permute_47.reshape(4, 64, 384)
        permute_47 = None
        qkv_bias_10 = (
            l_self_modules_features_modules_5_modules_6_modules_attn_modules_qkv_parameters_bias_.clone()
        )
        l_self_modules_features_modules_5_modules_6_modules_attn_modules_qkv_parameters_bias_ = (
            None
        )
        getitem_69 = qkv_bias_10[slice(384, 768, None)]
        zero__10 = getitem_69.zero_()
        getitem_69 = zero__10 = None
        qkv_20 = torch._C._nn.linear(
            x_130,
            l_self_modules_features_modules_5_modules_6_modules_attn_modules_qkv_parameters_weight_,
            qkv_bias_10,
        )
        x_130 = l_self_modules_features_modules_5_modules_6_modules_attn_modules_qkv_parameters_weight_ = (qkv_bias_10) = (
            None
        )
        reshape_46 = qkv_20.reshape(4, 64, 3, 12, 32)
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
            l_self_modules_features_modules_5_modules_6_modules_attn_parameters_logit_scale_,
            max=4.605170185988092,
        )
        l_self_modules_features_modules_5_modules_6_modules_attn_parameters_logit_scale_ = (
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
        x_131 = transpose_21.reshape(4, 64, 384)
        transpose_21 = None
        x_132 = torch._C._nn.linear(
            x_131,
            l_self_modules_features_modules_5_modules_6_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_attn_modules_proj_parameters_bias_,
        )
        x_131 = l_self_modules_features_modules_5_modules_6_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_attn_modules_proj_parameters_bias_ = (None)
        x_133 = torch.nn.functional.dropout(x_132, p=0.0, training=False)
        x_132 = None
        x_134 = x_133.view(1, 2, 2, 8, 8, 384)
        x_133 = None
        permute_49 = x_134.permute(0, 1, 3, 2, 4, 5)
        x_134 = None
        x_135 = permute_49.reshape(1, 16, 16, 384)
        permute_49 = None
        getitem_73 = x_135[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_135 = None
        x_136 = getitem_73.contiguous()
        getitem_73 = None
        layer_norm_23 = torch.nn.functional.layer_norm(
            x_136,
            (384,),
            l_self_modules_features_modules_5_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_136 = (
            l_self_modules_features_modules_5_modules_6_modules_norm1_parameters_weight_
        ) = (
            l_self_modules_features_modules_5_modules_6_modules_norm1_parameters_bias_
        ) = None
        _log_api_usage_once_20 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_20 = None
        x_137 = x_127 + layer_norm_23
        x_127 = layer_norm_23 = None
        input_87 = torch._C._nn.linear(
            x_137,
            l_self_modules_features_modules_5_modules_6_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_6_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_mlp_modules_0_parameters_bias_ = (None)
        input_88 = torch._C._nn.gelu(input_87, approximate="none")
        input_87 = None
        input_89 = torch.nn.functional.dropout(input_88, 0.0, False, False)
        input_88 = None
        input_90 = torch._C._nn.linear(
            input_89,
            l_self_modules_features_modules_5_modules_6_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_mlp_modules_3_parameters_bias_,
        )
        input_89 = l_self_modules_features_modules_5_modules_6_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_mlp_modules_3_parameters_bias_ = (None)
        input_91 = torch.nn.functional.dropout(input_90, 0.0, False, False)
        input_90 = None
        layer_norm_24 = torch.nn.functional.layer_norm(
            input_91,
            (384,),
            l_self_modules_features_modules_5_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_norm2_parameters_bias_,
            1e-05,
        )
        input_91 = (
            l_self_modules_features_modules_5_modules_6_modules_norm2_parameters_weight_
        ) = (
            l_self_modules_features_modules_5_modules_6_modules_norm2_parameters_bias_
        ) = None
        _log_api_usage_once_21 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_21 = None
        x_138 = x_137 + layer_norm_24
        x_137 = layer_norm_24 = None
        input_92 = torch._C._nn.linear(
            l_self_modules_features_modules_5_modules_7_modules_attn_buffers_relative_coords_table_,
            l_self_modules_features_modules_5_modules_7_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_7_modules_attn_buffers_relative_coords_table_ = l_self_modules_features_modules_5_modules_7_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = (None)
        input_93 = torch.nn.functional.relu(input_92, inplace=True)
        input_92 = None
        input_94 = torch._C._nn.linear(
            input_93,
            l_self_modules_features_modules_5_modules_7_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_,
            None,
        )
        input_93 = l_self_modules_features_modules_5_modules_7_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = (None)
        view_59 = input_94.view(-1, 12)
        input_94 = None
        relative_position_bias_44 = view_59[
            l_self_modules_features_modules_5_modules_7_modules_attn_buffers_relative_position_index_
        ]
        view_59 = l_self_modules_features_modules_5_modules_7_modules_attn_buffers_relative_position_index_ = (None)
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
        x_139 = torch._C._nn.pad(x_138, (0, 0, 0, 2, 0, 2), "constant", None)
        x_140 = torch.roll(x_139, shifts=(-4, -4), dims=(1, 2))
        x_139 = None
        x_141 = x_140.view(1, 2, 8, 2, 8, 384)
        x_140 = None
        permute_51 = x_141.permute(0, 1, 3, 2, 4, 5)
        x_141 = None
        x_142 = permute_51.reshape(4, 64, 384)
        permute_51 = None
        qkv_bias_11 = (
            l_self_modules_features_modules_5_modules_7_modules_attn_modules_qkv_parameters_bias_.clone()
        )
        l_self_modules_features_modules_5_modules_7_modules_attn_modules_qkv_parameters_bias_ = (
            None
        )
        getitem_75 = qkv_bias_11[slice(384, 768, None)]
        zero__11 = getitem_75.zero_()
        getitem_75 = zero__11 = None
        qkv_22 = torch._C._nn.linear(
            x_142,
            l_self_modules_features_modules_5_modules_7_modules_attn_modules_qkv_parameters_weight_,
            qkv_bias_11,
        )
        l_self_modules_features_modules_5_modules_7_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_11
        ) = None
        reshape_50 = qkv_22.reshape(4, 64, 3, 12, 32)
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
            l_self_modules_features_modules_5_modules_7_modules_attn_parameters_logit_scale_,
            max=4.605170185988092,
        )
        l_self_modules_features_modules_5_modules_7_modules_attn_parameters_logit_scale_ = (
            None
        )
        logit_scale_11 = clamp_11.exp()
        clamp_11 = None
        attn_71 = attn_70 * logit_scale_11
        attn_70 = logit_scale_11 = None
        attn_72 = attn_71 + relative_position_bias_47
        attn_71 = relative_position_bias_47 = None
        attn_mask_25 = x_142.new_zeros((16, 16))
        x_142 = None
        attn_mask_25[(slice(0, -8, None), slice(0, -8, None))] = 0
        setitem_45 = attn_mask_25
        setitem_45 = None
        attn_mask_25[(slice(0, -8, None), slice(-8, -4, None))] = 1
        setitem_46 = attn_mask_25
        setitem_46 = None
        attn_mask_25[(slice(0, -8, None), slice(-4, None, None))] = 2
        setitem_47 = attn_mask_25
        setitem_47 = None
        attn_mask_25[(slice(-8, -4, None), slice(0, -8, None))] = 3
        setitem_48 = attn_mask_25
        setitem_48 = None
        attn_mask_25[(slice(-8, -4, None), slice(-8, -4, None))] = 4
        setitem_49 = attn_mask_25
        setitem_49 = None
        attn_mask_25[(slice(-8, -4, None), slice(-4, None, None))] = 5
        setitem_50 = attn_mask_25
        setitem_50 = None
        attn_mask_25[(slice(-4, None, None), slice(0, -8, None))] = 6
        setitem_51 = attn_mask_25
        setitem_51 = None
        attn_mask_25[(slice(-4, None, None), slice(-8, -4, None))] = 7
        setitem_52 = attn_mask_25
        setitem_52 = None
        attn_mask_25[(slice(-4, None, None), slice(-4, None, None))] = 8
        setitem_53 = attn_mask_25
        setitem_53 = None
        attn_mask_26 = attn_mask_25.view(2, 8, 2, 8)
        attn_mask_25 = None
        permute_53 = attn_mask_26.permute(0, 2, 1, 3)
        attn_mask_26 = None
        attn_mask_27 = permute_53.reshape(4, 64)
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
        attn_73 = attn_72.view(1, 4, 12, 64, 64)
        attn_72 = None
        unsqueeze_34 = attn_mask_29.unsqueeze(1)
        attn_mask_29 = None
        unsqueeze_35 = unsqueeze_34.unsqueeze(0)
        unsqueeze_34 = None
        attn_74 = attn_73 + unsqueeze_35
        attn_73 = unsqueeze_35 = None
        attn_75 = attn_74.view(-1, 12, 64, 64)
        attn_74 = None
        attn_76 = torch.nn.functional.softmax(attn_75, dim=-1)
        attn_75 = None
        attn_77 = torch.nn.functional.dropout(attn_76, p=0.0, training=False)
        attn_76 = None
        matmul_23 = attn_77.matmul(v_11)
        attn_77 = v_11 = None
        transpose_23 = matmul_23.transpose(1, 2)
        matmul_23 = None
        x_143 = transpose_23.reshape(4, 64, 384)
        transpose_23 = None
        x_144 = torch._C._nn.linear(
            x_143,
            l_self_modules_features_modules_5_modules_7_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_attn_modules_proj_parameters_bias_,
        )
        x_143 = l_self_modules_features_modules_5_modules_7_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_attn_modules_proj_parameters_bias_ = (None)
        x_145 = torch.nn.functional.dropout(x_144, p=0.0, training=False)
        x_144 = None
        x_146 = x_145.view(1, 2, 2, 8, 8, 384)
        x_145 = None
        permute_54 = x_146.permute(0, 1, 3, 2, 4, 5)
        x_146 = None
        x_147 = permute_54.reshape(1, 16, 16, 384)
        permute_54 = None
        x_148 = torch.roll(x_147, shifts=(4, 4), dims=(1, 2))
        x_147 = None
        getitem_79 = x_148[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_148 = None
        x_149 = getitem_79.contiguous()
        getitem_79 = None
        layer_norm_25 = torch.nn.functional.layer_norm(
            x_149,
            (384,),
            l_self_modules_features_modules_5_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_149 = (
            l_self_modules_features_modules_5_modules_7_modules_norm1_parameters_weight_
        ) = (
            l_self_modules_features_modules_5_modules_7_modules_norm1_parameters_bias_
        ) = None
        _log_api_usage_once_22 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_22 = None
        x_150 = x_138 + layer_norm_25
        x_138 = layer_norm_25 = None
        input_95 = torch._C._nn.linear(
            x_150,
            l_self_modules_features_modules_5_modules_7_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_7_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_mlp_modules_0_parameters_bias_ = (None)
        input_96 = torch._C._nn.gelu(input_95, approximate="none")
        input_95 = None
        input_97 = torch.nn.functional.dropout(input_96, 0.0, False, False)
        input_96 = None
        input_98 = torch._C._nn.linear(
            input_97,
            l_self_modules_features_modules_5_modules_7_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_mlp_modules_3_parameters_bias_,
        )
        input_97 = l_self_modules_features_modules_5_modules_7_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_mlp_modules_3_parameters_bias_ = (None)
        input_99 = torch.nn.functional.dropout(input_98, 0.0, False, False)
        input_98 = None
        layer_norm_26 = torch.nn.functional.layer_norm(
            input_99,
            (384,),
            l_self_modules_features_modules_5_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_norm2_parameters_bias_,
            1e-05,
        )
        input_99 = (
            l_self_modules_features_modules_5_modules_7_modules_norm2_parameters_weight_
        ) = (
            l_self_modules_features_modules_5_modules_7_modules_norm2_parameters_bias_
        ) = None
        _log_api_usage_once_23 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_23 = None
        x_151 = x_150 + layer_norm_26
        x_150 = layer_norm_26 = None
        input_100 = torch._C._nn.linear(
            l_self_modules_features_modules_5_modules_8_modules_attn_buffers_relative_coords_table_,
            l_self_modules_features_modules_5_modules_8_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_8_modules_attn_buffers_relative_coords_table_ = l_self_modules_features_modules_5_modules_8_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_8_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = (None)
        input_101 = torch.nn.functional.relu(input_100, inplace=True)
        input_100 = None
        input_102 = torch._C._nn.linear(
            input_101,
            l_self_modules_features_modules_5_modules_8_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_,
            None,
        )
        input_101 = l_self_modules_features_modules_5_modules_8_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = (None)
        view_66 = input_102.view(-1, 12)
        input_102 = None
        relative_position_bias_48 = view_66[
            l_self_modules_features_modules_5_modules_8_modules_attn_buffers_relative_position_index_
        ]
        view_66 = l_self_modules_features_modules_5_modules_8_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_49 = relative_position_bias_48.view(64, 64, -1)
        relative_position_bias_48 = None
        permute_55 = relative_position_bias_49.permute(2, 0, 1)
        relative_position_bias_49 = None
        contiguous_24 = permute_55.contiguous()
        permute_55 = None
        relative_position_bias_50 = contiguous_24.unsqueeze(0)
        contiguous_24 = None
        sigmoid_12 = torch.sigmoid(relative_position_bias_50)
        relative_position_bias_50 = None
        relative_position_bias_51 = 16 * sigmoid_12
        sigmoid_12 = None
        x_152 = torch._C._nn.pad(x_151, (0, 0, 0, 2, 0, 2), "constant", None)
        x_153 = x_152.view(1, 2, 8, 2, 8, 384)
        x_152 = None
        permute_56 = x_153.permute(0, 1, 3, 2, 4, 5)
        x_153 = None
        x_154 = permute_56.reshape(4, 64, 384)
        permute_56 = None
        qkv_bias_12 = (
            l_self_modules_features_modules_5_modules_8_modules_attn_modules_qkv_parameters_bias_.clone()
        )
        l_self_modules_features_modules_5_modules_8_modules_attn_modules_qkv_parameters_bias_ = (
            None
        )
        getitem_81 = qkv_bias_12[slice(384, 768, None)]
        zero__12 = getitem_81.zero_()
        getitem_81 = zero__12 = None
        qkv_24 = torch._C._nn.linear(
            x_154,
            l_self_modules_features_modules_5_modules_8_modules_attn_modules_qkv_parameters_weight_,
            qkv_bias_12,
        )
        x_154 = l_self_modules_features_modules_5_modules_8_modules_attn_modules_qkv_parameters_weight_ = (qkv_bias_12) = (
            None
        )
        reshape_55 = qkv_24.reshape(4, 64, 3, 12, 32)
        qkv_24 = None
        qkv_25 = reshape_55.permute(2, 0, 3, 1, 4)
        reshape_55 = None
        q_12 = qkv_25[0]
        k_12 = qkv_25[1]
        v_12 = qkv_25[2]
        qkv_25 = None
        normalize_24 = torch.nn.functional.normalize(q_12, dim=-1)
        q_12 = None
        normalize_25 = torch.nn.functional.normalize(k_12, dim=-1)
        k_12 = None
        transpose_24 = normalize_25.transpose(-2, -1)
        normalize_25 = None
        attn_78 = normalize_24 @ transpose_24
        normalize_24 = transpose_24 = None
        clamp_12 = torch.clamp(
            l_self_modules_features_modules_5_modules_8_modules_attn_parameters_logit_scale_,
            max=4.605170185988092,
        )
        l_self_modules_features_modules_5_modules_8_modules_attn_parameters_logit_scale_ = (
            None
        )
        logit_scale_12 = clamp_12.exp()
        clamp_12 = None
        attn_79 = attn_78 * logit_scale_12
        attn_78 = logit_scale_12 = None
        attn_80 = attn_79 + relative_position_bias_51
        attn_79 = relative_position_bias_51 = None
        attn_81 = torch.nn.functional.softmax(attn_80, dim=-1)
        attn_80 = None
        attn_82 = torch.nn.functional.dropout(attn_81, p=0.0, training=False)
        attn_81 = None
        matmul_25 = attn_82.matmul(v_12)
        attn_82 = v_12 = None
        transpose_25 = matmul_25.transpose(1, 2)
        matmul_25 = None
        x_155 = transpose_25.reshape(4, 64, 384)
        transpose_25 = None
        x_156 = torch._C._nn.linear(
            x_155,
            l_self_modules_features_modules_5_modules_8_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_attn_modules_proj_parameters_bias_,
        )
        x_155 = l_self_modules_features_modules_5_modules_8_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_5_modules_8_modules_attn_modules_proj_parameters_bias_ = (None)
        x_157 = torch.nn.functional.dropout(x_156, p=0.0, training=False)
        x_156 = None
        x_158 = x_157.view(1, 2, 2, 8, 8, 384)
        x_157 = None
        permute_58 = x_158.permute(0, 1, 3, 2, 4, 5)
        x_158 = None
        x_159 = permute_58.reshape(1, 16, 16, 384)
        permute_58 = None
        getitem_85 = x_159[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_159 = None
        x_160 = getitem_85.contiguous()
        getitem_85 = None
        layer_norm_27 = torch.nn.functional.layer_norm(
            x_160,
            (384,),
            l_self_modules_features_modules_5_modules_8_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_160 = (
            l_self_modules_features_modules_5_modules_8_modules_norm1_parameters_weight_
        ) = (
            l_self_modules_features_modules_5_modules_8_modules_norm1_parameters_bias_
        ) = None
        _log_api_usage_once_24 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_24 = None
        x_161 = x_151 + layer_norm_27
        x_151 = layer_norm_27 = None
        input_103 = torch._C._nn.linear(
            x_161,
            l_self_modules_features_modules_5_modules_8_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_8_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_8_modules_mlp_modules_0_parameters_bias_ = (None)
        input_104 = torch._C._nn.gelu(input_103, approximate="none")
        input_103 = None
        input_105 = torch.nn.functional.dropout(input_104, 0.0, False, False)
        input_104 = None
        input_106 = torch._C._nn.linear(
            input_105,
            l_self_modules_features_modules_5_modules_8_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_mlp_modules_3_parameters_bias_,
        )
        input_105 = l_self_modules_features_modules_5_modules_8_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_8_modules_mlp_modules_3_parameters_bias_ = (None)
        input_107 = torch.nn.functional.dropout(input_106, 0.0, False, False)
        input_106 = None
        layer_norm_28 = torch.nn.functional.layer_norm(
            input_107,
            (384,),
            l_self_modules_features_modules_5_modules_8_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_norm2_parameters_bias_,
            1e-05,
        )
        input_107 = (
            l_self_modules_features_modules_5_modules_8_modules_norm2_parameters_weight_
        ) = (
            l_self_modules_features_modules_5_modules_8_modules_norm2_parameters_bias_
        ) = None
        _log_api_usage_once_25 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_25 = None
        x_162 = x_161 + layer_norm_28
        x_161 = layer_norm_28 = None
        input_108 = torch._C._nn.linear(
            l_self_modules_features_modules_5_modules_9_modules_attn_buffers_relative_coords_table_,
            l_self_modules_features_modules_5_modules_9_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_9_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_9_modules_attn_buffers_relative_coords_table_ = l_self_modules_features_modules_5_modules_9_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_9_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = (None)
        input_109 = torch.nn.functional.relu(input_108, inplace=True)
        input_108 = None
        input_110 = torch._C._nn.linear(
            input_109,
            l_self_modules_features_modules_5_modules_9_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_,
            None,
        )
        input_109 = l_self_modules_features_modules_5_modules_9_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = (None)
        view_70 = input_110.view(-1, 12)
        input_110 = None
        relative_position_bias_52 = view_70[
            l_self_modules_features_modules_5_modules_9_modules_attn_buffers_relative_position_index_
        ]
        view_70 = l_self_modules_features_modules_5_modules_9_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_53 = relative_position_bias_52.view(64, 64, -1)
        relative_position_bias_52 = None
        permute_59 = relative_position_bias_53.permute(2, 0, 1)
        relative_position_bias_53 = None
        contiguous_26 = permute_59.contiguous()
        permute_59 = None
        relative_position_bias_54 = contiguous_26.unsqueeze(0)
        contiguous_26 = None
        sigmoid_13 = torch.sigmoid(relative_position_bias_54)
        relative_position_bias_54 = None
        relative_position_bias_55 = 16 * sigmoid_13
        sigmoid_13 = None
        x_163 = torch._C._nn.pad(x_162, (0, 0, 0, 2, 0, 2), "constant", None)
        x_164 = torch.roll(x_163, shifts=(-4, -4), dims=(1, 2))
        x_163 = None
        x_165 = x_164.view(1, 2, 8, 2, 8, 384)
        x_164 = None
        permute_60 = x_165.permute(0, 1, 3, 2, 4, 5)
        x_165 = None
        x_166 = permute_60.reshape(4, 64, 384)
        permute_60 = None
        qkv_bias_13 = (
            l_self_modules_features_modules_5_modules_9_modules_attn_modules_qkv_parameters_bias_.clone()
        )
        l_self_modules_features_modules_5_modules_9_modules_attn_modules_qkv_parameters_bias_ = (
            None
        )
        getitem_87 = qkv_bias_13[slice(384, 768, None)]
        zero__13 = getitem_87.zero_()
        getitem_87 = zero__13 = None
        qkv_26 = torch._C._nn.linear(
            x_166,
            l_self_modules_features_modules_5_modules_9_modules_attn_modules_qkv_parameters_weight_,
            qkv_bias_13,
        )
        l_self_modules_features_modules_5_modules_9_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_13
        ) = None
        reshape_59 = qkv_26.reshape(4, 64, 3, 12, 32)
        qkv_26 = None
        qkv_27 = reshape_59.permute(2, 0, 3, 1, 4)
        reshape_59 = None
        q_13 = qkv_27[0]
        k_13 = qkv_27[1]
        v_13 = qkv_27[2]
        qkv_27 = None
        normalize_26 = torch.nn.functional.normalize(q_13, dim=-1)
        q_13 = None
        normalize_27 = torch.nn.functional.normalize(k_13, dim=-1)
        k_13 = None
        transpose_26 = normalize_27.transpose(-2, -1)
        normalize_27 = None
        attn_83 = normalize_26 @ transpose_26
        normalize_26 = transpose_26 = None
        clamp_13 = torch.clamp(
            l_self_modules_features_modules_5_modules_9_modules_attn_parameters_logit_scale_,
            max=4.605170185988092,
        )
        l_self_modules_features_modules_5_modules_9_modules_attn_parameters_logit_scale_ = (
            None
        )
        logit_scale_13 = clamp_13.exp()
        clamp_13 = None
        attn_84 = attn_83 * logit_scale_13
        attn_83 = logit_scale_13 = None
        attn_85 = attn_84 + relative_position_bias_55
        attn_84 = relative_position_bias_55 = None
        attn_mask_30 = x_166.new_zeros((16, 16))
        x_166 = None
        attn_mask_30[(slice(0, -8, None), slice(0, -8, None))] = 0
        setitem_54 = attn_mask_30
        setitem_54 = None
        attn_mask_30[(slice(0, -8, None), slice(-8, -4, None))] = 1
        setitem_55 = attn_mask_30
        setitem_55 = None
        attn_mask_30[(slice(0, -8, None), slice(-4, None, None))] = 2
        setitem_56 = attn_mask_30
        setitem_56 = None
        attn_mask_30[(slice(-8, -4, None), slice(0, -8, None))] = 3
        setitem_57 = attn_mask_30
        setitem_57 = None
        attn_mask_30[(slice(-8, -4, None), slice(-8, -4, None))] = 4
        setitem_58 = attn_mask_30
        setitem_58 = None
        attn_mask_30[(slice(-8, -4, None), slice(-4, None, None))] = 5
        setitem_59 = attn_mask_30
        setitem_59 = None
        attn_mask_30[(slice(-4, None, None), slice(0, -8, None))] = 6
        setitem_60 = attn_mask_30
        setitem_60 = None
        attn_mask_30[(slice(-4, None, None), slice(-8, -4, None))] = 7
        setitem_61 = attn_mask_30
        setitem_61 = None
        attn_mask_30[(slice(-4, None, None), slice(-4, None, None))] = 8
        setitem_62 = attn_mask_30
        setitem_62 = None
        attn_mask_31 = attn_mask_30.view(2, 8, 2, 8)
        attn_mask_30 = None
        permute_62 = attn_mask_31.permute(0, 2, 1, 3)
        attn_mask_31 = None
        attn_mask_32 = permute_62.reshape(4, 64)
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
        attn_86 = attn_85.view(1, 4, 12, 64, 64)
        attn_85 = None
        unsqueeze_40 = attn_mask_34.unsqueeze(1)
        attn_mask_34 = None
        unsqueeze_41 = unsqueeze_40.unsqueeze(0)
        unsqueeze_40 = None
        attn_87 = attn_86 + unsqueeze_41
        attn_86 = unsqueeze_41 = None
        attn_88 = attn_87.view(-1, 12, 64, 64)
        attn_87 = None
        attn_89 = torch.nn.functional.softmax(attn_88, dim=-1)
        attn_88 = None
        attn_90 = torch.nn.functional.dropout(attn_89, p=0.0, training=False)
        attn_89 = None
        matmul_27 = attn_90.matmul(v_13)
        attn_90 = v_13 = None
        transpose_27 = matmul_27.transpose(1, 2)
        matmul_27 = None
        x_167 = transpose_27.reshape(4, 64, 384)
        transpose_27 = None
        x_168 = torch._C._nn.linear(
            x_167,
            l_self_modules_features_modules_5_modules_9_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_5_modules_9_modules_attn_modules_proj_parameters_bias_,
        )
        x_167 = l_self_modules_features_modules_5_modules_9_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_5_modules_9_modules_attn_modules_proj_parameters_bias_ = (None)
        x_169 = torch.nn.functional.dropout(x_168, p=0.0, training=False)
        x_168 = None
        x_170 = x_169.view(1, 2, 2, 8, 8, 384)
        x_169 = None
        permute_63 = x_170.permute(0, 1, 3, 2, 4, 5)
        x_170 = None
        x_171 = permute_63.reshape(1, 16, 16, 384)
        permute_63 = None
        x_172 = torch.roll(x_171, shifts=(4, 4), dims=(1, 2))
        x_171 = None
        getitem_91 = x_172[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_172 = None
        x_173 = getitem_91.contiguous()
        getitem_91 = None
        layer_norm_29 = torch.nn.functional.layer_norm(
            x_173,
            (384,),
            l_self_modules_features_modules_5_modules_9_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_5_modules_9_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_173 = (
            l_self_modules_features_modules_5_modules_9_modules_norm1_parameters_weight_
        ) = (
            l_self_modules_features_modules_5_modules_9_modules_norm1_parameters_bias_
        ) = None
        _log_api_usage_once_26 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_26 = None
        x_174 = x_162 + layer_norm_29
        x_162 = layer_norm_29 = None
        input_111 = torch._C._nn.linear(
            x_174,
            l_self_modules_features_modules_5_modules_9_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_9_modules_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_9_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_9_modules_mlp_modules_0_parameters_bias_ = (None)
        input_112 = torch._C._nn.gelu(input_111, approximate="none")
        input_111 = None
        input_113 = torch.nn.functional.dropout(input_112, 0.0, False, False)
        input_112 = None
        input_114 = torch._C._nn.linear(
            input_113,
            l_self_modules_features_modules_5_modules_9_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_9_modules_mlp_modules_3_parameters_bias_,
        )
        input_113 = l_self_modules_features_modules_5_modules_9_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_9_modules_mlp_modules_3_parameters_bias_ = (None)
        input_115 = torch.nn.functional.dropout(input_114, 0.0, False, False)
        input_114 = None
        layer_norm_30 = torch.nn.functional.layer_norm(
            input_115,
            (384,),
            l_self_modules_features_modules_5_modules_9_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_5_modules_9_modules_norm2_parameters_bias_,
            1e-05,
        )
        input_115 = (
            l_self_modules_features_modules_5_modules_9_modules_norm2_parameters_weight_
        ) = (
            l_self_modules_features_modules_5_modules_9_modules_norm2_parameters_bias_
        ) = None
        _log_api_usage_once_27 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_27 = None
        x_175 = x_174 + layer_norm_30
        x_174 = layer_norm_30 = None
        input_116 = torch._C._nn.linear(
            l_self_modules_features_modules_5_modules_10_modules_attn_buffers_relative_coords_table_,
            l_self_modules_features_modules_5_modules_10_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_10_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_10_modules_attn_buffers_relative_coords_table_ = l_self_modules_features_modules_5_modules_10_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_10_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = (None)
        input_117 = torch.nn.functional.relu(input_116, inplace=True)
        input_116 = None
        input_118 = torch._C._nn.linear(
            input_117,
            l_self_modules_features_modules_5_modules_10_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_,
            None,
        )
        input_117 = l_self_modules_features_modules_5_modules_10_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = (None)
        view_77 = input_118.view(-1, 12)
        input_118 = None
        relative_position_bias_56 = view_77[
            l_self_modules_features_modules_5_modules_10_modules_attn_buffers_relative_position_index_
        ]
        view_77 = l_self_modules_features_modules_5_modules_10_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_57 = relative_position_bias_56.view(64, 64, -1)
        relative_position_bias_56 = None
        permute_64 = relative_position_bias_57.permute(2, 0, 1)
        relative_position_bias_57 = None
        contiguous_28 = permute_64.contiguous()
        permute_64 = None
        relative_position_bias_58 = contiguous_28.unsqueeze(0)
        contiguous_28 = None
        sigmoid_14 = torch.sigmoid(relative_position_bias_58)
        relative_position_bias_58 = None
        relative_position_bias_59 = 16 * sigmoid_14
        sigmoid_14 = None
        x_176 = torch._C._nn.pad(x_175, (0, 0, 0, 2, 0, 2), "constant", None)
        x_177 = x_176.view(1, 2, 8, 2, 8, 384)
        x_176 = None
        permute_65 = x_177.permute(0, 1, 3, 2, 4, 5)
        x_177 = None
        x_178 = permute_65.reshape(4, 64, 384)
        permute_65 = None
        qkv_bias_14 = (
            l_self_modules_features_modules_5_modules_10_modules_attn_modules_qkv_parameters_bias_.clone()
        )
        l_self_modules_features_modules_5_modules_10_modules_attn_modules_qkv_parameters_bias_ = (
            None
        )
        getitem_93 = qkv_bias_14[slice(384, 768, None)]
        zero__14 = getitem_93.zero_()
        getitem_93 = zero__14 = None
        qkv_28 = torch._C._nn.linear(
            x_178,
            l_self_modules_features_modules_5_modules_10_modules_attn_modules_qkv_parameters_weight_,
            qkv_bias_14,
        )
        x_178 = l_self_modules_features_modules_5_modules_10_modules_attn_modules_qkv_parameters_weight_ = (qkv_bias_14) = (
            None
        )
        reshape_64 = qkv_28.reshape(4, 64, 3, 12, 32)
        qkv_28 = None
        qkv_29 = reshape_64.permute(2, 0, 3, 1, 4)
        reshape_64 = None
        q_14 = qkv_29[0]
        k_14 = qkv_29[1]
        v_14 = qkv_29[2]
        qkv_29 = None
        normalize_28 = torch.nn.functional.normalize(q_14, dim=-1)
        q_14 = None
        normalize_29 = torch.nn.functional.normalize(k_14, dim=-1)
        k_14 = None
        transpose_28 = normalize_29.transpose(-2, -1)
        normalize_29 = None
        attn_91 = normalize_28 @ transpose_28
        normalize_28 = transpose_28 = None
        clamp_14 = torch.clamp(
            l_self_modules_features_modules_5_modules_10_modules_attn_parameters_logit_scale_,
            max=4.605170185988092,
        )
        l_self_modules_features_modules_5_modules_10_modules_attn_parameters_logit_scale_ = (
            None
        )
        logit_scale_14 = clamp_14.exp()
        clamp_14 = None
        attn_92 = attn_91 * logit_scale_14
        attn_91 = logit_scale_14 = None
        attn_93 = attn_92 + relative_position_bias_59
        attn_92 = relative_position_bias_59 = None
        attn_94 = torch.nn.functional.softmax(attn_93, dim=-1)
        attn_93 = None
        attn_95 = torch.nn.functional.dropout(attn_94, p=0.0, training=False)
        attn_94 = None
        matmul_29 = attn_95.matmul(v_14)
        attn_95 = v_14 = None
        transpose_29 = matmul_29.transpose(1, 2)
        matmul_29 = None
        x_179 = transpose_29.reshape(4, 64, 384)
        transpose_29 = None
        x_180 = torch._C._nn.linear(
            x_179,
            l_self_modules_features_modules_5_modules_10_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_5_modules_10_modules_attn_modules_proj_parameters_bias_,
        )
        x_179 = l_self_modules_features_modules_5_modules_10_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_5_modules_10_modules_attn_modules_proj_parameters_bias_ = (None)
        x_181 = torch.nn.functional.dropout(x_180, p=0.0, training=False)
        x_180 = None
        x_182 = x_181.view(1, 2, 2, 8, 8, 384)
        x_181 = None
        permute_67 = x_182.permute(0, 1, 3, 2, 4, 5)
        x_182 = None
        x_183 = permute_67.reshape(1, 16, 16, 384)
        permute_67 = None
        getitem_97 = x_183[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_183 = None
        x_184 = getitem_97.contiguous()
        getitem_97 = None
        layer_norm_31 = torch.nn.functional.layer_norm(
            x_184,
            (384,),
            l_self_modules_features_modules_5_modules_10_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_5_modules_10_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_184 = l_self_modules_features_modules_5_modules_10_modules_norm1_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_10_modules_norm1_parameters_bias_
        ) = None
        _log_api_usage_once_28 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_28 = None
        x_185 = x_175 + layer_norm_31
        x_175 = layer_norm_31 = None
        input_119 = torch._C._nn.linear(
            x_185,
            l_self_modules_features_modules_5_modules_10_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_10_modules_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_10_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_10_modules_mlp_modules_0_parameters_bias_ = (None)
        input_120 = torch._C._nn.gelu(input_119, approximate="none")
        input_119 = None
        input_121 = torch.nn.functional.dropout(input_120, 0.0, False, False)
        input_120 = None
        input_122 = torch._C._nn.linear(
            input_121,
            l_self_modules_features_modules_5_modules_10_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_10_modules_mlp_modules_3_parameters_bias_,
        )
        input_121 = l_self_modules_features_modules_5_modules_10_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_10_modules_mlp_modules_3_parameters_bias_ = (None)
        input_123 = torch.nn.functional.dropout(input_122, 0.0, False, False)
        input_122 = None
        layer_norm_32 = torch.nn.functional.layer_norm(
            input_123,
            (384,),
            l_self_modules_features_modules_5_modules_10_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_5_modules_10_modules_norm2_parameters_bias_,
            1e-05,
        )
        input_123 = l_self_modules_features_modules_5_modules_10_modules_norm2_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_10_modules_norm2_parameters_bias_
        ) = None
        _log_api_usage_once_29 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_29 = None
        x_186 = x_185 + layer_norm_32
        x_185 = layer_norm_32 = None
        input_124 = torch._C._nn.linear(
            l_self_modules_features_modules_5_modules_11_modules_attn_buffers_relative_coords_table_,
            l_self_modules_features_modules_5_modules_11_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_11_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_11_modules_attn_buffers_relative_coords_table_ = l_self_modules_features_modules_5_modules_11_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_11_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = (None)
        input_125 = torch.nn.functional.relu(input_124, inplace=True)
        input_124 = None
        input_126 = torch._C._nn.linear(
            input_125,
            l_self_modules_features_modules_5_modules_11_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_,
            None,
        )
        input_125 = l_self_modules_features_modules_5_modules_11_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = (None)
        view_81 = input_126.view(-1, 12)
        input_126 = None
        relative_position_bias_60 = view_81[
            l_self_modules_features_modules_5_modules_11_modules_attn_buffers_relative_position_index_
        ]
        view_81 = l_self_modules_features_modules_5_modules_11_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_61 = relative_position_bias_60.view(64, 64, -1)
        relative_position_bias_60 = None
        permute_68 = relative_position_bias_61.permute(2, 0, 1)
        relative_position_bias_61 = None
        contiguous_30 = permute_68.contiguous()
        permute_68 = None
        relative_position_bias_62 = contiguous_30.unsqueeze(0)
        contiguous_30 = None
        sigmoid_15 = torch.sigmoid(relative_position_bias_62)
        relative_position_bias_62 = None
        relative_position_bias_63 = 16 * sigmoid_15
        sigmoid_15 = None
        x_187 = torch._C._nn.pad(x_186, (0, 0, 0, 2, 0, 2), "constant", None)
        x_188 = torch.roll(x_187, shifts=(-4, -4), dims=(1, 2))
        x_187 = None
        x_189 = x_188.view(1, 2, 8, 2, 8, 384)
        x_188 = None
        permute_69 = x_189.permute(0, 1, 3, 2, 4, 5)
        x_189 = None
        x_190 = permute_69.reshape(4, 64, 384)
        permute_69 = None
        qkv_bias_15 = (
            l_self_modules_features_modules_5_modules_11_modules_attn_modules_qkv_parameters_bias_.clone()
        )
        l_self_modules_features_modules_5_modules_11_modules_attn_modules_qkv_parameters_bias_ = (
            None
        )
        getitem_99 = qkv_bias_15[slice(384, 768, None)]
        zero__15 = getitem_99.zero_()
        getitem_99 = zero__15 = None
        qkv_30 = torch._C._nn.linear(
            x_190,
            l_self_modules_features_modules_5_modules_11_modules_attn_modules_qkv_parameters_weight_,
            qkv_bias_15,
        )
        l_self_modules_features_modules_5_modules_11_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_15
        ) = None
        reshape_68 = qkv_30.reshape(4, 64, 3, 12, 32)
        qkv_30 = None
        qkv_31 = reshape_68.permute(2, 0, 3, 1, 4)
        reshape_68 = None
        q_15 = qkv_31[0]
        k_15 = qkv_31[1]
        v_15 = qkv_31[2]
        qkv_31 = None
        normalize_30 = torch.nn.functional.normalize(q_15, dim=-1)
        q_15 = None
        normalize_31 = torch.nn.functional.normalize(k_15, dim=-1)
        k_15 = None
        transpose_30 = normalize_31.transpose(-2, -1)
        normalize_31 = None
        attn_96 = normalize_30 @ transpose_30
        normalize_30 = transpose_30 = None
        clamp_15 = torch.clamp(
            l_self_modules_features_modules_5_modules_11_modules_attn_parameters_logit_scale_,
            max=4.605170185988092,
        )
        l_self_modules_features_modules_5_modules_11_modules_attn_parameters_logit_scale_ = (
            None
        )
        logit_scale_15 = clamp_15.exp()
        clamp_15 = None
        attn_97 = attn_96 * logit_scale_15
        attn_96 = logit_scale_15 = None
        attn_98 = attn_97 + relative_position_bias_63
        attn_97 = relative_position_bias_63 = None
        attn_mask_35 = x_190.new_zeros((16, 16))
        x_190 = None
        attn_mask_35[(slice(0, -8, None), slice(0, -8, None))] = 0
        setitem_63 = attn_mask_35
        setitem_63 = None
        attn_mask_35[(slice(0, -8, None), slice(-8, -4, None))] = 1
        setitem_64 = attn_mask_35
        setitem_64 = None
        attn_mask_35[(slice(0, -8, None), slice(-4, None, None))] = 2
        setitem_65 = attn_mask_35
        setitem_65 = None
        attn_mask_35[(slice(-8, -4, None), slice(0, -8, None))] = 3
        setitem_66 = attn_mask_35
        setitem_66 = None
        attn_mask_35[(slice(-8, -4, None), slice(-8, -4, None))] = 4
        setitem_67 = attn_mask_35
        setitem_67 = None
        attn_mask_35[(slice(-8, -4, None), slice(-4, None, None))] = 5
        setitem_68 = attn_mask_35
        setitem_68 = None
        attn_mask_35[(slice(-4, None, None), slice(0, -8, None))] = 6
        setitem_69 = attn_mask_35
        setitem_69 = None
        attn_mask_35[(slice(-4, None, None), slice(-8, -4, None))] = 7
        setitem_70 = attn_mask_35
        setitem_70 = None
        attn_mask_35[(slice(-4, None, None), slice(-4, None, None))] = 8
        setitem_71 = attn_mask_35
        setitem_71 = None
        attn_mask_36 = attn_mask_35.view(2, 8, 2, 8)
        attn_mask_35 = None
        permute_71 = attn_mask_36.permute(0, 2, 1, 3)
        attn_mask_36 = None
        attn_mask_37 = permute_71.reshape(4, 64)
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
        attn_99 = attn_98.view(1, 4, 12, 64, 64)
        attn_98 = None
        unsqueeze_46 = attn_mask_39.unsqueeze(1)
        attn_mask_39 = None
        unsqueeze_47 = unsqueeze_46.unsqueeze(0)
        unsqueeze_46 = None
        attn_100 = attn_99 + unsqueeze_47
        attn_99 = unsqueeze_47 = None
        attn_101 = attn_100.view(-1, 12, 64, 64)
        attn_100 = None
        attn_102 = torch.nn.functional.softmax(attn_101, dim=-1)
        attn_101 = None
        attn_103 = torch.nn.functional.dropout(attn_102, p=0.0, training=False)
        attn_102 = None
        matmul_31 = attn_103.matmul(v_15)
        attn_103 = v_15 = None
        transpose_31 = matmul_31.transpose(1, 2)
        matmul_31 = None
        x_191 = transpose_31.reshape(4, 64, 384)
        transpose_31 = None
        x_192 = torch._C._nn.linear(
            x_191,
            l_self_modules_features_modules_5_modules_11_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_5_modules_11_modules_attn_modules_proj_parameters_bias_,
        )
        x_191 = l_self_modules_features_modules_5_modules_11_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_5_modules_11_modules_attn_modules_proj_parameters_bias_ = (None)
        x_193 = torch.nn.functional.dropout(x_192, p=0.0, training=False)
        x_192 = None
        x_194 = x_193.view(1, 2, 2, 8, 8, 384)
        x_193 = None
        permute_72 = x_194.permute(0, 1, 3, 2, 4, 5)
        x_194 = None
        x_195 = permute_72.reshape(1, 16, 16, 384)
        permute_72 = None
        x_196 = torch.roll(x_195, shifts=(4, 4), dims=(1, 2))
        x_195 = None
        getitem_103 = x_196[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_196 = None
        x_197 = getitem_103.contiguous()
        getitem_103 = None
        layer_norm_33 = torch.nn.functional.layer_norm(
            x_197,
            (384,),
            l_self_modules_features_modules_5_modules_11_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_5_modules_11_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_197 = l_self_modules_features_modules_5_modules_11_modules_norm1_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_11_modules_norm1_parameters_bias_
        ) = None
        _log_api_usage_once_30 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_30 = None
        x_198 = x_186 + layer_norm_33
        x_186 = layer_norm_33 = None
        input_127 = torch._C._nn.linear(
            x_198,
            l_self_modules_features_modules_5_modules_11_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_11_modules_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_11_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_11_modules_mlp_modules_0_parameters_bias_ = (None)
        input_128 = torch._C._nn.gelu(input_127, approximate="none")
        input_127 = None
        input_129 = torch.nn.functional.dropout(input_128, 0.0, False, False)
        input_128 = None
        input_130 = torch._C._nn.linear(
            input_129,
            l_self_modules_features_modules_5_modules_11_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_11_modules_mlp_modules_3_parameters_bias_,
        )
        input_129 = l_self_modules_features_modules_5_modules_11_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_11_modules_mlp_modules_3_parameters_bias_ = (None)
        input_131 = torch.nn.functional.dropout(input_130, 0.0, False, False)
        input_130 = None
        layer_norm_34 = torch.nn.functional.layer_norm(
            input_131,
            (384,),
            l_self_modules_features_modules_5_modules_11_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_5_modules_11_modules_norm2_parameters_bias_,
            1e-05,
        )
        input_131 = l_self_modules_features_modules_5_modules_11_modules_norm2_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_11_modules_norm2_parameters_bias_
        ) = None
        _log_api_usage_once_31 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_31 = None
        x_199 = x_198 + layer_norm_34
        x_198 = layer_norm_34 = None
        input_132 = torch._C._nn.linear(
            l_self_modules_features_modules_5_modules_12_modules_attn_buffers_relative_coords_table_,
            l_self_modules_features_modules_5_modules_12_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_12_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_12_modules_attn_buffers_relative_coords_table_ = l_self_modules_features_modules_5_modules_12_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_12_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = (None)
        input_133 = torch.nn.functional.relu(input_132, inplace=True)
        input_132 = None
        input_134 = torch._C._nn.linear(
            input_133,
            l_self_modules_features_modules_5_modules_12_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_,
            None,
        )
        input_133 = l_self_modules_features_modules_5_modules_12_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = (None)
        view_88 = input_134.view(-1, 12)
        input_134 = None
        relative_position_bias_64 = view_88[
            l_self_modules_features_modules_5_modules_12_modules_attn_buffers_relative_position_index_
        ]
        view_88 = l_self_modules_features_modules_5_modules_12_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_65 = relative_position_bias_64.view(64, 64, -1)
        relative_position_bias_64 = None
        permute_73 = relative_position_bias_65.permute(2, 0, 1)
        relative_position_bias_65 = None
        contiguous_32 = permute_73.contiguous()
        permute_73 = None
        relative_position_bias_66 = contiguous_32.unsqueeze(0)
        contiguous_32 = None
        sigmoid_16 = torch.sigmoid(relative_position_bias_66)
        relative_position_bias_66 = None
        relative_position_bias_67 = 16 * sigmoid_16
        sigmoid_16 = None
        x_200 = torch._C._nn.pad(x_199, (0, 0, 0, 2, 0, 2), "constant", None)
        x_201 = x_200.view(1, 2, 8, 2, 8, 384)
        x_200 = None
        permute_74 = x_201.permute(0, 1, 3, 2, 4, 5)
        x_201 = None
        x_202 = permute_74.reshape(4, 64, 384)
        permute_74 = None
        qkv_bias_16 = (
            l_self_modules_features_modules_5_modules_12_modules_attn_modules_qkv_parameters_bias_.clone()
        )
        l_self_modules_features_modules_5_modules_12_modules_attn_modules_qkv_parameters_bias_ = (
            None
        )
        getitem_105 = qkv_bias_16[slice(384, 768, None)]
        zero__16 = getitem_105.zero_()
        getitem_105 = zero__16 = None
        qkv_32 = torch._C._nn.linear(
            x_202,
            l_self_modules_features_modules_5_modules_12_modules_attn_modules_qkv_parameters_weight_,
            qkv_bias_16,
        )
        x_202 = l_self_modules_features_modules_5_modules_12_modules_attn_modules_qkv_parameters_weight_ = (qkv_bias_16) = (
            None
        )
        reshape_73 = qkv_32.reshape(4, 64, 3, 12, 32)
        qkv_32 = None
        qkv_33 = reshape_73.permute(2, 0, 3, 1, 4)
        reshape_73 = None
        q_16 = qkv_33[0]
        k_16 = qkv_33[1]
        v_16 = qkv_33[2]
        qkv_33 = None
        normalize_32 = torch.nn.functional.normalize(q_16, dim=-1)
        q_16 = None
        normalize_33 = torch.nn.functional.normalize(k_16, dim=-1)
        k_16 = None
        transpose_32 = normalize_33.transpose(-2, -1)
        normalize_33 = None
        attn_104 = normalize_32 @ transpose_32
        normalize_32 = transpose_32 = None
        clamp_16 = torch.clamp(
            l_self_modules_features_modules_5_modules_12_modules_attn_parameters_logit_scale_,
            max=4.605170185988092,
        )
        l_self_modules_features_modules_5_modules_12_modules_attn_parameters_logit_scale_ = (
            None
        )
        logit_scale_16 = clamp_16.exp()
        clamp_16 = None
        attn_105 = attn_104 * logit_scale_16
        attn_104 = logit_scale_16 = None
        attn_106 = attn_105 + relative_position_bias_67
        attn_105 = relative_position_bias_67 = None
        attn_107 = torch.nn.functional.softmax(attn_106, dim=-1)
        attn_106 = None
        attn_108 = torch.nn.functional.dropout(attn_107, p=0.0, training=False)
        attn_107 = None
        matmul_33 = attn_108.matmul(v_16)
        attn_108 = v_16 = None
        transpose_33 = matmul_33.transpose(1, 2)
        matmul_33 = None
        x_203 = transpose_33.reshape(4, 64, 384)
        transpose_33 = None
        x_204 = torch._C._nn.linear(
            x_203,
            l_self_modules_features_modules_5_modules_12_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_5_modules_12_modules_attn_modules_proj_parameters_bias_,
        )
        x_203 = l_self_modules_features_modules_5_modules_12_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_5_modules_12_modules_attn_modules_proj_parameters_bias_ = (None)
        x_205 = torch.nn.functional.dropout(x_204, p=0.0, training=False)
        x_204 = None
        x_206 = x_205.view(1, 2, 2, 8, 8, 384)
        x_205 = None
        permute_76 = x_206.permute(0, 1, 3, 2, 4, 5)
        x_206 = None
        x_207 = permute_76.reshape(1, 16, 16, 384)
        permute_76 = None
        getitem_109 = x_207[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_207 = None
        x_208 = getitem_109.contiguous()
        getitem_109 = None
        layer_norm_35 = torch.nn.functional.layer_norm(
            x_208,
            (384,),
            l_self_modules_features_modules_5_modules_12_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_5_modules_12_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_208 = l_self_modules_features_modules_5_modules_12_modules_norm1_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_12_modules_norm1_parameters_bias_
        ) = None
        _log_api_usage_once_32 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_32 = None
        x_209 = x_199 + layer_norm_35
        x_199 = layer_norm_35 = None
        input_135 = torch._C._nn.linear(
            x_209,
            l_self_modules_features_modules_5_modules_12_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_12_modules_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_12_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_12_modules_mlp_modules_0_parameters_bias_ = (None)
        input_136 = torch._C._nn.gelu(input_135, approximate="none")
        input_135 = None
        input_137 = torch.nn.functional.dropout(input_136, 0.0, False, False)
        input_136 = None
        input_138 = torch._C._nn.linear(
            input_137,
            l_self_modules_features_modules_5_modules_12_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_12_modules_mlp_modules_3_parameters_bias_,
        )
        input_137 = l_self_modules_features_modules_5_modules_12_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_12_modules_mlp_modules_3_parameters_bias_ = (None)
        input_139 = torch.nn.functional.dropout(input_138, 0.0, False, False)
        input_138 = None
        layer_norm_36 = torch.nn.functional.layer_norm(
            input_139,
            (384,),
            l_self_modules_features_modules_5_modules_12_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_5_modules_12_modules_norm2_parameters_bias_,
            1e-05,
        )
        input_139 = l_self_modules_features_modules_5_modules_12_modules_norm2_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_12_modules_norm2_parameters_bias_
        ) = None
        _log_api_usage_once_33 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_33 = None
        x_210 = x_209 + layer_norm_36
        x_209 = layer_norm_36 = None
        input_140 = torch._C._nn.linear(
            l_self_modules_features_modules_5_modules_13_modules_attn_buffers_relative_coords_table_,
            l_self_modules_features_modules_5_modules_13_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_13_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_13_modules_attn_buffers_relative_coords_table_ = l_self_modules_features_modules_5_modules_13_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_13_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = (None)
        input_141 = torch.nn.functional.relu(input_140, inplace=True)
        input_140 = None
        input_142 = torch._C._nn.linear(
            input_141,
            l_self_modules_features_modules_5_modules_13_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_,
            None,
        )
        input_141 = l_self_modules_features_modules_5_modules_13_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = (None)
        view_92 = input_142.view(-1, 12)
        input_142 = None
        relative_position_bias_68 = view_92[
            l_self_modules_features_modules_5_modules_13_modules_attn_buffers_relative_position_index_
        ]
        view_92 = l_self_modules_features_modules_5_modules_13_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_69 = relative_position_bias_68.view(64, 64, -1)
        relative_position_bias_68 = None
        permute_77 = relative_position_bias_69.permute(2, 0, 1)
        relative_position_bias_69 = None
        contiguous_34 = permute_77.contiguous()
        permute_77 = None
        relative_position_bias_70 = contiguous_34.unsqueeze(0)
        contiguous_34 = None
        sigmoid_17 = torch.sigmoid(relative_position_bias_70)
        relative_position_bias_70 = None
        relative_position_bias_71 = 16 * sigmoid_17
        sigmoid_17 = None
        x_211 = torch._C._nn.pad(x_210, (0, 0, 0, 2, 0, 2), "constant", None)
        x_212 = torch.roll(x_211, shifts=(-4, -4), dims=(1, 2))
        x_211 = None
        x_213 = x_212.view(1, 2, 8, 2, 8, 384)
        x_212 = None
        permute_78 = x_213.permute(0, 1, 3, 2, 4, 5)
        x_213 = None
        x_214 = permute_78.reshape(4, 64, 384)
        permute_78 = None
        qkv_bias_17 = (
            l_self_modules_features_modules_5_modules_13_modules_attn_modules_qkv_parameters_bias_.clone()
        )
        l_self_modules_features_modules_5_modules_13_modules_attn_modules_qkv_parameters_bias_ = (
            None
        )
        getitem_111 = qkv_bias_17[slice(384, 768, None)]
        zero__17 = getitem_111.zero_()
        getitem_111 = zero__17 = None
        qkv_34 = torch._C._nn.linear(
            x_214,
            l_self_modules_features_modules_5_modules_13_modules_attn_modules_qkv_parameters_weight_,
            qkv_bias_17,
        )
        l_self_modules_features_modules_5_modules_13_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_17
        ) = None
        reshape_77 = qkv_34.reshape(4, 64, 3, 12, 32)
        qkv_34 = None
        qkv_35 = reshape_77.permute(2, 0, 3, 1, 4)
        reshape_77 = None
        q_17 = qkv_35[0]
        k_17 = qkv_35[1]
        v_17 = qkv_35[2]
        qkv_35 = None
        normalize_34 = torch.nn.functional.normalize(q_17, dim=-1)
        q_17 = None
        normalize_35 = torch.nn.functional.normalize(k_17, dim=-1)
        k_17 = None
        transpose_34 = normalize_35.transpose(-2, -1)
        normalize_35 = None
        attn_109 = normalize_34 @ transpose_34
        normalize_34 = transpose_34 = None
        clamp_17 = torch.clamp(
            l_self_modules_features_modules_5_modules_13_modules_attn_parameters_logit_scale_,
            max=4.605170185988092,
        )
        l_self_modules_features_modules_5_modules_13_modules_attn_parameters_logit_scale_ = (
            None
        )
        logit_scale_17 = clamp_17.exp()
        clamp_17 = None
        attn_110 = attn_109 * logit_scale_17
        attn_109 = logit_scale_17 = None
        attn_111 = attn_110 + relative_position_bias_71
        attn_110 = relative_position_bias_71 = None
        attn_mask_40 = x_214.new_zeros((16, 16))
        x_214 = None
        attn_mask_40[(slice(0, -8, None), slice(0, -8, None))] = 0
        setitem_72 = attn_mask_40
        setitem_72 = None
        attn_mask_40[(slice(0, -8, None), slice(-8, -4, None))] = 1
        setitem_73 = attn_mask_40
        setitem_73 = None
        attn_mask_40[(slice(0, -8, None), slice(-4, None, None))] = 2
        setitem_74 = attn_mask_40
        setitem_74 = None
        attn_mask_40[(slice(-8, -4, None), slice(0, -8, None))] = 3
        setitem_75 = attn_mask_40
        setitem_75 = None
        attn_mask_40[(slice(-8, -4, None), slice(-8, -4, None))] = 4
        setitem_76 = attn_mask_40
        setitem_76 = None
        attn_mask_40[(slice(-8, -4, None), slice(-4, None, None))] = 5
        setitem_77 = attn_mask_40
        setitem_77 = None
        attn_mask_40[(slice(-4, None, None), slice(0, -8, None))] = 6
        setitem_78 = attn_mask_40
        setitem_78 = None
        attn_mask_40[(slice(-4, None, None), slice(-8, -4, None))] = 7
        setitem_79 = attn_mask_40
        setitem_79 = None
        attn_mask_40[(slice(-4, None, None), slice(-4, None, None))] = 8
        setitem_80 = attn_mask_40
        setitem_80 = None
        attn_mask_41 = attn_mask_40.view(2, 8, 2, 8)
        attn_mask_40 = None
        permute_80 = attn_mask_41.permute(0, 2, 1, 3)
        attn_mask_41 = None
        attn_mask_42 = permute_80.reshape(4, 64)
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
        attn_112 = attn_111.view(1, 4, 12, 64, 64)
        attn_111 = None
        unsqueeze_52 = attn_mask_44.unsqueeze(1)
        attn_mask_44 = None
        unsqueeze_53 = unsqueeze_52.unsqueeze(0)
        unsqueeze_52 = None
        attn_113 = attn_112 + unsqueeze_53
        attn_112 = unsqueeze_53 = None
        attn_114 = attn_113.view(-1, 12, 64, 64)
        attn_113 = None
        attn_115 = torch.nn.functional.softmax(attn_114, dim=-1)
        attn_114 = None
        attn_116 = torch.nn.functional.dropout(attn_115, p=0.0, training=False)
        attn_115 = None
        matmul_35 = attn_116.matmul(v_17)
        attn_116 = v_17 = None
        transpose_35 = matmul_35.transpose(1, 2)
        matmul_35 = None
        x_215 = transpose_35.reshape(4, 64, 384)
        transpose_35 = None
        x_216 = torch._C._nn.linear(
            x_215,
            l_self_modules_features_modules_5_modules_13_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_5_modules_13_modules_attn_modules_proj_parameters_bias_,
        )
        x_215 = l_self_modules_features_modules_5_modules_13_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_5_modules_13_modules_attn_modules_proj_parameters_bias_ = (None)
        x_217 = torch.nn.functional.dropout(x_216, p=0.0, training=False)
        x_216 = None
        x_218 = x_217.view(1, 2, 2, 8, 8, 384)
        x_217 = None
        permute_81 = x_218.permute(0, 1, 3, 2, 4, 5)
        x_218 = None
        x_219 = permute_81.reshape(1, 16, 16, 384)
        permute_81 = None
        x_220 = torch.roll(x_219, shifts=(4, 4), dims=(1, 2))
        x_219 = None
        getitem_115 = x_220[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_220 = None
        x_221 = getitem_115.contiguous()
        getitem_115 = None
        layer_norm_37 = torch.nn.functional.layer_norm(
            x_221,
            (384,),
            l_self_modules_features_modules_5_modules_13_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_5_modules_13_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_221 = l_self_modules_features_modules_5_modules_13_modules_norm1_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_13_modules_norm1_parameters_bias_
        ) = None
        _log_api_usage_once_34 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_34 = None
        x_222 = x_210 + layer_norm_37
        x_210 = layer_norm_37 = None
        input_143 = torch._C._nn.linear(
            x_222,
            l_self_modules_features_modules_5_modules_13_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_13_modules_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_13_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_13_modules_mlp_modules_0_parameters_bias_ = (None)
        input_144 = torch._C._nn.gelu(input_143, approximate="none")
        input_143 = None
        input_145 = torch.nn.functional.dropout(input_144, 0.0, False, False)
        input_144 = None
        input_146 = torch._C._nn.linear(
            input_145,
            l_self_modules_features_modules_5_modules_13_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_13_modules_mlp_modules_3_parameters_bias_,
        )
        input_145 = l_self_modules_features_modules_5_modules_13_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_13_modules_mlp_modules_3_parameters_bias_ = (None)
        input_147 = torch.nn.functional.dropout(input_146, 0.0, False, False)
        input_146 = None
        layer_norm_38 = torch.nn.functional.layer_norm(
            input_147,
            (384,),
            l_self_modules_features_modules_5_modules_13_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_5_modules_13_modules_norm2_parameters_bias_,
            1e-05,
        )
        input_147 = l_self_modules_features_modules_5_modules_13_modules_norm2_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_13_modules_norm2_parameters_bias_
        ) = None
        _log_api_usage_once_35 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_35 = None
        x_223 = x_222 + layer_norm_38
        x_222 = layer_norm_38 = None
        input_148 = torch._C._nn.linear(
            l_self_modules_features_modules_5_modules_14_modules_attn_buffers_relative_coords_table_,
            l_self_modules_features_modules_5_modules_14_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_14_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_14_modules_attn_buffers_relative_coords_table_ = l_self_modules_features_modules_5_modules_14_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_14_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = (None)
        input_149 = torch.nn.functional.relu(input_148, inplace=True)
        input_148 = None
        input_150 = torch._C._nn.linear(
            input_149,
            l_self_modules_features_modules_5_modules_14_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_,
            None,
        )
        input_149 = l_self_modules_features_modules_5_modules_14_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = (None)
        view_99 = input_150.view(-1, 12)
        input_150 = None
        relative_position_bias_72 = view_99[
            l_self_modules_features_modules_5_modules_14_modules_attn_buffers_relative_position_index_
        ]
        view_99 = l_self_modules_features_modules_5_modules_14_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_73 = relative_position_bias_72.view(64, 64, -1)
        relative_position_bias_72 = None
        permute_82 = relative_position_bias_73.permute(2, 0, 1)
        relative_position_bias_73 = None
        contiguous_36 = permute_82.contiguous()
        permute_82 = None
        relative_position_bias_74 = contiguous_36.unsqueeze(0)
        contiguous_36 = None
        sigmoid_18 = torch.sigmoid(relative_position_bias_74)
        relative_position_bias_74 = None
        relative_position_bias_75 = 16 * sigmoid_18
        sigmoid_18 = None
        x_224 = torch._C._nn.pad(x_223, (0, 0, 0, 2, 0, 2), "constant", None)
        x_225 = x_224.view(1, 2, 8, 2, 8, 384)
        x_224 = None
        permute_83 = x_225.permute(0, 1, 3, 2, 4, 5)
        x_225 = None
        x_226 = permute_83.reshape(4, 64, 384)
        permute_83 = None
        qkv_bias_18 = (
            l_self_modules_features_modules_5_modules_14_modules_attn_modules_qkv_parameters_bias_.clone()
        )
        l_self_modules_features_modules_5_modules_14_modules_attn_modules_qkv_parameters_bias_ = (
            None
        )
        getitem_117 = qkv_bias_18[slice(384, 768, None)]
        zero__18 = getitem_117.zero_()
        getitem_117 = zero__18 = None
        qkv_36 = torch._C._nn.linear(
            x_226,
            l_self_modules_features_modules_5_modules_14_modules_attn_modules_qkv_parameters_weight_,
            qkv_bias_18,
        )
        x_226 = l_self_modules_features_modules_5_modules_14_modules_attn_modules_qkv_parameters_weight_ = (qkv_bias_18) = (
            None
        )
        reshape_82 = qkv_36.reshape(4, 64, 3, 12, 32)
        qkv_36 = None
        qkv_37 = reshape_82.permute(2, 0, 3, 1, 4)
        reshape_82 = None
        q_18 = qkv_37[0]
        k_18 = qkv_37[1]
        v_18 = qkv_37[2]
        qkv_37 = None
        normalize_36 = torch.nn.functional.normalize(q_18, dim=-1)
        q_18 = None
        normalize_37 = torch.nn.functional.normalize(k_18, dim=-1)
        k_18 = None
        transpose_36 = normalize_37.transpose(-2, -1)
        normalize_37 = None
        attn_117 = normalize_36 @ transpose_36
        normalize_36 = transpose_36 = None
        clamp_18 = torch.clamp(
            l_self_modules_features_modules_5_modules_14_modules_attn_parameters_logit_scale_,
            max=4.605170185988092,
        )
        l_self_modules_features_modules_5_modules_14_modules_attn_parameters_logit_scale_ = (
            None
        )
        logit_scale_18 = clamp_18.exp()
        clamp_18 = None
        attn_118 = attn_117 * logit_scale_18
        attn_117 = logit_scale_18 = None
        attn_119 = attn_118 + relative_position_bias_75
        attn_118 = relative_position_bias_75 = None
        attn_120 = torch.nn.functional.softmax(attn_119, dim=-1)
        attn_119 = None
        attn_121 = torch.nn.functional.dropout(attn_120, p=0.0, training=False)
        attn_120 = None
        matmul_37 = attn_121.matmul(v_18)
        attn_121 = v_18 = None
        transpose_37 = matmul_37.transpose(1, 2)
        matmul_37 = None
        x_227 = transpose_37.reshape(4, 64, 384)
        transpose_37 = None
        x_228 = torch._C._nn.linear(
            x_227,
            l_self_modules_features_modules_5_modules_14_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_5_modules_14_modules_attn_modules_proj_parameters_bias_,
        )
        x_227 = l_self_modules_features_modules_5_modules_14_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_5_modules_14_modules_attn_modules_proj_parameters_bias_ = (None)
        x_229 = torch.nn.functional.dropout(x_228, p=0.0, training=False)
        x_228 = None
        x_230 = x_229.view(1, 2, 2, 8, 8, 384)
        x_229 = None
        permute_85 = x_230.permute(0, 1, 3, 2, 4, 5)
        x_230 = None
        x_231 = permute_85.reshape(1, 16, 16, 384)
        permute_85 = None
        getitem_121 = x_231[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_231 = None
        x_232 = getitem_121.contiguous()
        getitem_121 = None
        layer_norm_39 = torch.nn.functional.layer_norm(
            x_232,
            (384,),
            l_self_modules_features_modules_5_modules_14_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_5_modules_14_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_232 = l_self_modules_features_modules_5_modules_14_modules_norm1_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_14_modules_norm1_parameters_bias_
        ) = None
        _log_api_usage_once_36 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_36 = None
        x_233 = x_223 + layer_norm_39
        x_223 = layer_norm_39 = None
        input_151 = torch._C._nn.linear(
            x_233,
            l_self_modules_features_modules_5_modules_14_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_14_modules_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_14_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_14_modules_mlp_modules_0_parameters_bias_ = (None)
        input_152 = torch._C._nn.gelu(input_151, approximate="none")
        input_151 = None
        input_153 = torch.nn.functional.dropout(input_152, 0.0, False, False)
        input_152 = None
        input_154 = torch._C._nn.linear(
            input_153,
            l_self_modules_features_modules_5_modules_14_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_14_modules_mlp_modules_3_parameters_bias_,
        )
        input_153 = l_self_modules_features_modules_5_modules_14_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_14_modules_mlp_modules_3_parameters_bias_ = (None)
        input_155 = torch.nn.functional.dropout(input_154, 0.0, False, False)
        input_154 = None
        layer_norm_40 = torch.nn.functional.layer_norm(
            input_155,
            (384,),
            l_self_modules_features_modules_5_modules_14_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_5_modules_14_modules_norm2_parameters_bias_,
            1e-05,
        )
        input_155 = l_self_modules_features_modules_5_modules_14_modules_norm2_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_14_modules_norm2_parameters_bias_
        ) = None
        _log_api_usage_once_37 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_37 = None
        x_234 = x_233 + layer_norm_40
        x_233 = layer_norm_40 = None
        input_156 = torch._C._nn.linear(
            l_self_modules_features_modules_5_modules_15_modules_attn_buffers_relative_coords_table_,
            l_self_modules_features_modules_5_modules_15_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_15_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_15_modules_attn_buffers_relative_coords_table_ = l_self_modules_features_modules_5_modules_15_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_15_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = (None)
        input_157 = torch.nn.functional.relu(input_156, inplace=True)
        input_156 = None
        input_158 = torch._C._nn.linear(
            input_157,
            l_self_modules_features_modules_5_modules_15_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_,
            None,
        )
        input_157 = l_self_modules_features_modules_5_modules_15_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = (None)
        view_103 = input_158.view(-1, 12)
        input_158 = None
        relative_position_bias_76 = view_103[
            l_self_modules_features_modules_5_modules_15_modules_attn_buffers_relative_position_index_
        ]
        view_103 = l_self_modules_features_modules_5_modules_15_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_77 = relative_position_bias_76.view(64, 64, -1)
        relative_position_bias_76 = None
        permute_86 = relative_position_bias_77.permute(2, 0, 1)
        relative_position_bias_77 = None
        contiguous_38 = permute_86.contiguous()
        permute_86 = None
        relative_position_bias_78 = contiguous_38.unsqueeze(0)
        contiguous_38 = None
        sigmoid_19 = torch.sigmoid(relative_position_bias_78)
        relative_position_bias_78 = None
        relative_position_bias_79 = 16 * sigmoid_19
        sigmoid_19 = None
        x_235 = torch._C._nn.pad(x_234, (0, 0, 0, 2, 0, 2), "constant", None)
        x_236 = torch.roll(x_235, shifts=(-4, -4), dims=(1, 2))
        x_235 = None
        x_237 = x_236.view(1, 2, 8, 2, 8, 384)
        x_236 = None
        permute_87 = x_237.permute(0, 1, 3, 2, 4, 5)
        x_237 = None
        x_238 = permute_87.reshape(4, 64, 384)
        permute_87 = None
        qkv_bias_19 = (
            l_self_modules_features_modules_5_modules_15_modules_attn_modules_qkv_parameters_bias_.clone()
        )
        l_self_modules_features_modules_5_modules_15_modules_attn_modules_qkv_parameters_bias_ = (
            None
        )
        getitem_123 = qkv_bias_19[slice(384, 768, None)]
        zero__19 = getitem_123.zero_()
        getitem_123 = zero__19 = None
        qkv_38 = torch._C._nn.linear(
            x_238,
            l_self_modules_features_modules_5_modules_15_modules_attn_modules_qkv_parameters_weight_,
            qkv_bias_19,
        )
        l_self_modules_features_modules_5_modules_15_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_19
        ) = None
        reshape_86 = qkv_38.reshape(4, 64, 3, 12, 32)
        qkv_38 = None
        qkv_39 = reshape_86.permute(2, 0, 3, 1, 4)
        reshape_86 = None
        q_19 = qkv_39[0]
        k_19 = qkv_39[1]
        v_19 = qkv_39[2]
        qkv_39 = None
        normalize_38 = torch.nn.functional.normalize(q_19, dim=-1)
        q_19 = None
        normalize_39 = torch.nn.functional.normalize(k_19, dim=-1)
        k_19 = None
        transpose_38 = normalize_39.transpose(-2, -1)
        normalize_39 = None
        attn_122 = normalize_38 @ transpose_38
        normalize_38 = transpose_38 = None
        clamp_19 = torch.clamp(
            l_self_modules_features_modules_5_modules_15_modules_attn_parameters_logit_scale_,
            max=4.605170185988092,
        )
        l_self_modules_features_modules_5_modules_15_modules_attn_parameters_logit_scale_ = (
            None
        )
        logit_scale_19 = clamp_19.exp()
        clamp_19 = None
        attn_123 = attn_122 * logit_scale_19
        attn_122 = logit_scale_19 = None
        attn_124 = attn_123 + relative_position_bias_79
        attn_123 = relative_position_bias_79 = None
        attn_mask_45 = x_238.new_zeros((16, 16))
        x_238 = None
        attn_mask_45[(slice(0, -8, None), slice(0, -8, None))] = 0
        setitem_81 = attn_mask_45
        setitem_81 = None
        attn_mask_45[(slice(0, -8, None), slice(-8, -4, None))] = 1
        setitem_82 = attn_mask_45
        setitem_82 = None
        attn_mask_45[(slice(0, -8, None), slice(-4, None, None))] = 2
        setitem_83 = attn_mask_45
        setitem_83 = None
        attn_mask_45[(slice(-8, -4, None), slice(0, -8, None))] = 3
        setitem_84 = attn_mask_45
        setitem_84 = None
        attn_mask_45[(slice(-8, -4, None), slice(-8, -4, None))] = 4
        setitem_85 = attn_mask_45
        setitem_85 = None
        attn_mask_45[(slice(-8, -4, None), slice(-4, None, None))] = 5
        setitem_86 = attn_mask_45
        setitem_86 = None
        attn_mask_45[(slice(-4, None, None), slice(0, -8, None))] = 6
        setitem_87 = attn_mask_45
        setitem_87 = None
        attn_mask_45[(slice(-4, None, None), slice(-8, -4, None))] = 7
        setitem_88 = attn_mask_45
        setitem_88 = None
        attn_mask_45[(slice(-4, None, None), slice(-4, None, None))] = 8
        setitem_89 = attn_mask_45
        setitem_89 = None
        attn_mask_46 = attn_mask_45.view(2, 8, 2, 8)
        attn_mask_45 = None
        permute_89 = attn_mask_46.permute(0, 2, 1, 3)
        attn_mask_46 = None
        attn_mask_47 = permute_89.reshape(4, 64)
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
        attn_125 = attn_124.view(1, 4, 12, 64, 64)
        attn_124 = None
        unsqueeze_58 = attn_mask_49.unsqueeze(1)
        attn_mask_49 = None
        unsqueeze_59 = unsqueeze_58.unsqueeze(0)
        unsqueeze_58 = None
        attn_126 = attn_125 + unsqueeze_59
        attn_125 = unsqueeze_59 = None
        attn_127 = attn_126.view(-1, 12, 64, 64)
        attn_126 = None
        attn_128 = torch.nn.functional.softmax(attn_127, dim=-1)
        attn_127 = None
        attn_129 = torch.nn.functional.dropout(attn_128, p=0.0, training=False)
        attn_128 = None
        matmul_39 = attn_129.matmul(v_19)
        attn_129 = v_19 = None
        transpose_39 = matmul_39.transpose(1, 2)
        matmul_39 = None
        x_239 = transpose_39.reshape(4, 64, 384)
        transpose_39 = None
        x_240 = torch._C._nn.linear(
            x_239,
            l_self_modules_features_modules_5_modules_15_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_5_modules_15_modules_attn_modules_proj_parameters_bias_,
        )
        x_239 = l_self_modules_features_modules_5_modules_15_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_5_modules_15_modules_attn_modules_proj_parameters_bias_ = (None)
        x_241 = torch.nn.functional.dropout(x_240, p=0.0, training=False)
        x_240 = None
        x_242 = x_241.view(1, 2, 2, 8, 8, 384)
        x_241 = None
        permute_90 = x_242.permute(0, 1, 3, 2, 4, 5)
        x_242 = None
        x_243 = permute_90.reshape(1, 16, 16, 384)
        permute_90 = None
        x_244 = torch.roll(x_243, shifts=(4, 4), dims=(1, 2))
        x_243 = None
        getitem_127 = x_244[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_244 = None
        x_245 = getitem_127.contiguous()
        getitem_127 = None
        layer_norm_41 = torch.nn.functional.layer_norm(
            x_245,
            (384,),
            l_self_modules_features_modules_5_modules_15_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_5_modules_15_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_245 = l_self_modules_features_modules_5_modules_15_modules_norm1_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_15_modules_norm1_parameters_bias_
        ) = None
        _log_api_usage_once_38 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_38 = None
        x_246 = x_234 + layer_norm_41
        x_234 = layer_norm_41 = None
        input_159 = torch._C._nn.linear(
            x_246,
            l_self_modules_features_modules_5_modules_15_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_15_modules_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_15_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_15_modules_mlp_modules_0_parameters_bias_ = (None)
        input_160 = torch._C._nn.gelu(input_159, approximate="none")
        input_159 = None
        input_161 = torch.nn.functional.dropout(input_160, 0.0, False, False)
        input_160 = None
        input_162 = torch._C._nn.linear(
            input_161,
            l_self_modules_features_modules_5_modules_15_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_15_modules_mlp_modules_3_parameters_bias_,
        )
        input_161 = l_self_modules_features_modules_5_modules_15_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_15_modules_mlp_modules_3_parameters_bias_ = (None)
        input_163 = torch.nn.functional.dropout(input_162, 0.0, False, False)
        input_162 = None
        layer_norm_42 = torch.nn.functional.layer_norm(
            input_163,
            (384,),
            l_self_modules_features_modules_5_modules_15_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_5_modules_15_modules_norm2_parameters_bias_,
            1e-05,
        )
        input_163 = l_self_modules_features_modules_5_modules_15_modules_norm2_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_15_modules_norm2_parameters_bias_
        ) = None
        _log_api_usage_once_39 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_39 = None
        x_247 = x_246 + layer_norm_42
        x_246 = layer_norm_42 = None
        input_164 = torch._C._nn.linear(
            l_self_modules_features_modules_5_modules_16_modules_attn_buffers_relative_coords_table_,
            l_self_modules_features_modules_5_modules_16_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_16_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_16_modules_attn_buffers_relative_coords_table_ = l_self_modules_features_modules_5_modules_16_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_16_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = (None)
        input_165 = torch.nn.functional.relu(input_164, inplace=True)
        input_164 = None
        input_166 = torch._C._nn.linear(
            input_165,
            l_self_modules_features_modules_5_modules_16_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_,
            None,
        )
        input_165 = l_self_modules_features_modules_5_modules_16_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = (None)
        view_110 = input_166.view(-1, 12)
        input_166 = None
        relative_position_bias_80 = view_110[
            l_self_modules_features_modules_5_modules_16_modules_attn_buffers_relative_position_index_
        ]
        view_110 = l_self_modules_features_modules_5_modules_16_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_81 = relative_position_bias_80.view(64, 64, -1)
        relative_position_bias_80 = None
        permute_91 = relative_position_bias_81.permute(2, 0, 1)
        relative_position_bias_81 = None
        contiguous_40 = permute_91.contiguous()
        permute_91 = None
        relative_position_bias_82 = contiguous_40.unsqueeze(0)
        contiguous_40 = None
        sigmoid_20 = torch.sigmoid(relative_position_bias_82)
        relative_position_bias_82 = None
        relative_position_bias_83 = 16 * sigmoid_20
        sigmoid_20 = None
        x_248 = torch._C._nn.pad(x_247, (0, 0, 0, 2, 0, 2), "constant", None)
        x_249 = x_248.view(1, 2, 8, 2, 8, 384)
        x_248 = None
        permute_92 = x_249.permute(0, 1, 3, 2, 4, 5)
        x_249 = None
        x_250 = permute_92.reshape(4, 64, 384)
        permute_92 = None
        qkv_bias_20 = (
            l_self_modules_features_modules_5_modules_16_modules_attn_modules_qkv_parameters_bias_.clone()
        )
        l_self_modules_features_modules_5_modules_16_modules_attn_modules_qkv_parameters_bias_ = (
            None
        )
        getitem_129 = qkv_bias_20[slice(384, 768, None)]
        zero__20 = getitem_129.zero_()
        getitem_129 = zero__20 = None
        qkv_40 = torch._C._nn.linear(
            x_250,
            l_self_modules_features_modules_5_modules_16_modules_attn_modules_qkv_parameters_weight_,
            qkv_bias_20,
        )
        x_250 = l_self_modules_features_modules_5_modules_16_modules_attn_modules_qkv_parameters_weight_ = (qkv_bias_20) = (
            None
        )
        reshape_91 = qkv_40.reshape(4, 64, 3, 12, 32)
        qkv_40 = None
        qkv_41 = reshape_91.permute(2, 0, 3, 1, 4)
        reshape_91 = None
        q_20 = qkv_41[0]
        k_20 = qkv_41[1]
        v_20 = qkv_41[2]
        qkv_41 = None
        normalize_40 = torch.nn.functional.normalize(q_20, dim=-1)
        q_20 = None
        normalize_41 = torch.nn.functional.normalize(k_20, dim=-1)
        k_20 = None
        transpose_40 = normalize_41.transpose(-2, -1)
        normalize_41 = None
        attn_130 = normalize_40 @ transpose_40
        normalize_40 = transpose_40 = None
        clamp_20 = torch.clamp(
            l_self_modules_features_modules_5_modules_16_modules_attn_parameters_logit_scale_,
            max=4.605170185988092,
        )
        l_self_modules_features_modules_5_modules_16_modules_attn_parameters_logit_scale_ = (
            None
        )
        logit_scale_20 = clamp_20.exp()
        clamp_20 = None
        attn_131 = attn_130 * logit_scale_20
        attn_130 = logit_scale_20 = None
        attn_132 = attn_131 + relative_position_bias_83
        attn_131 = relative_position_bias_83 = None
        attn_133 = torch.nn.functional.softmax(attn_132, dim=-1)
        attn_132 = None
        attn_134 = torch.nn.functional.dropout(attn_133, p=0.0, training=False)
        attn_133 = None
        matmul_41 = attn_134.matmul(v_20)
        attn_134 = v_20 = None
        transpose_41 = matmul_41.transpose(1, 2)
        matmul_41 = None
        x_251 = transpose_41.reshape(4, 64, 384)
        transpose_41 = None
        x_252 = torch._C._nn.linear(
            x_251,
            l_self_modules_features_modules_5_modules_16_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_5_modules_16_modules_attn_modules_proj_parameters_bias_,
        )
        x_251 = l_self_modules_features_modules_5_modules_16_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_5_modules_16_modules_attn_modules_proj_parameters_bias_ = (None)
        x_253 = torch.nn.functional.dropout(x_252, p=0.0, training=False)
        x_252 = None
        x_254 = x_253.view(1, 2, 2, 8, 8, 384)
        x_253 = None
        permute_94 = x_254.permute(0, 1, 3, 2, 4, 5)
        x_254 = None
        x_255 = permute_94.reshape(1, 16, 16, 384)
        permute_94 = None
        getitem_133 = x_255[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_255 = None
        x_256 = getitem_133.contiguous()
        getitem_133 = None
        layer_norm_43 = torch.nn.functional.layer_norm(
            x_256,
            (384,),
            l_self_modules_features_modules_5_modules_16_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_5_modules_16_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_256 = l_self_modules_features_modules_5_modules_16_modules_norm1_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_16_modules_norm1_parameters_bias_
        ) = None
        _log_api_usage_once_40 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_40 = None
        x_257 = x_247 + layer_norm_43
        x_247 = layer_norm_43 = None
        input_167 = torch._C._nn.linear(
            x_257,
            l_self_modules_features_modules_5_modules_16_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_16_modules_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_16_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_16_modules_mlp_modules_0_parameters_bias_ = (None)
        input_168 = torch._C._nn.gelu(input_167, approximate="none")
        input_167 = None
        input_169 = torch.nn.functional.dropout(input_168, 0.0, False, False)
        input_168 = None
        input_170 = torch._C._nn.linear(
            input_169,
            l_self_modules_features_modules_5_modules_16_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_16_modules_mlp_modules_3_parameters_bias_,
        )
        input_169 = l_self_modules_features_modules_5_modules_16_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_16_modules_mlp_modules_3_parameters_bias_ = (None)
        input_171 = torch.nn.functional.dropout(input_170, 0.0, False, False)
        input_170 = None
        layer_norm_44 = torch.nn.functional.layer_norm(
            input_171,
            (384,),
            l_self_modules_features_modules_5_modules_16_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_5_modules_16_modules_norm2_parameters_bias_,
            1e-05,
        )
        input_171 = l_self_modules_features_modules_5_modules_16_modules_norm2_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_16_modules_norm2_parameters_bias_
        ) = None
        _log_api_usage_once_41 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_41 = None
        x_258 = x_257 + layer_norm_44
        x_257 = layer_norm_44 = None
        input_172 = torch._C._nn.linear(
            l_self_modules_features_modules_5_modules_17_modules_attn_buffers_relative_coords_table_,
            l_self_modules_features_modules_5_modules_17_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_17_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_17_modules_attn_buffers_relative_coords_table_ = l_self_modules_features_modules_5_modules_17_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_17_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = (None)
        input_173 = torch.nn.functional.relu(input_172, inplace=True)
        input_172 = None
        input_174 = torch._C._nn.linear(
            input_173,
            l_self_modules_features_modules_5_modules_17_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_,
            None,
        )
        input_173 = l_self_modules_features_modules_5_modules_17_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = (None)
        view_114 = input_174.view(-1, 12)
        input_174 = None
        relative_position_bias_84 = view_114[
            l_self_modules_features_modules_5_modules_17_modules_attn_buffers_relative_position_index_
        ]
        view_114 = l_self_modules_features_modules_5_modules_17_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_85 = relative_position_bias_84.view(64, 64, -1)
        relative_position_bias_84 = None
        permute_95 = relative_position_bias_85.permute(2, 0, 1)
        relative_position_bias_85 = None
        contiguous_42 = permute_95.contiguous()
        permute_95 = None
        relative_position_bias_86 = contiguous_42.unsqueeze(0)
        contiguous_42 = None
        sigmoid_21 = torch.sigmoid(relative_position_bias_86)
        relative_position_bias_86 = None
        relative_position_bias_87 = 16 * sigmoid_21
        sigmoid_21 = None
        x_259 = torch._C._nn.pad(x_258, (0, 0, 0, 2, 0, 2), "constant", None)
        x_260 = torch.roll(x_259, shifts=(-4, -4), dims=(1, 2))
        x_259 = None
        x_261 = x_260.view(1, 2, 8, 2, 8, 384)
        x_260 = None
        permute_96 = x_261.permute(0, 1, 3, 2, 4, 5)
        x_261 = None
        x_262 = permute_96.reshape(4, 64, 384)
        permute_96 = None
        qkv_bias_21 = (
            l_self_modules_features_modules_5_modules_17_modules_attn_modules_qkv_parameters_bias_.clone()
        )
        l_self_modules_features_modules_5_modules_17_modules_attn_modules_qkv_parameters_bias_ = (
            None
        )
        getitem_135 = qkv_bias_21[slice(384, 768, None)]
        zero__21 = getitem_135.zero_()
        getitem_135 = zero__21 = None
        qkv_42 = torch._C._nn.linear(
            x_262,
            l_self_modules_features_modules_5_modules_17_modules_attn_modules_qkv_parameters_weight_,
            qkv_bias_21,
        )
        l_self_modules_features_modules_5_modules_17_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_21
        ) = None
        reshape_95 = qkv_42.reshape(4, 64, 3, 12, 32)
        qkv_42 = None
        qkv_43 = reshape_95.permute(2, 0, 3, 1, 4)
        reshape_95 = None
        q_21 = qkv_43[0]
        k_21 = qkv_43[1]
        v_21 = qkv_43[2]
        qkv_43 = None
        normalize_42 = torch.nn.functional.normalize(q_21, dim=-1)
        q_21 = None
        normalize_43 = torch.nn.functional.normalize(k_21, dim=-1)
        k_21 = None
        transpose_42 = normalize_43.transpose(-2, -1)
        normalize_43 = None
        attn_135 = normalize_42 @ transpose_42
        normalize_42 = transpose_42 = None
        clamp_21 = torch.clamp(
            l_self_modules_features_modules_5_modules_17_modules_attn_parameters_logit_scale_,
            max=4.605170185988092,
        )
        l_self_modules_features_modules_5_modules_17_modules_attn_parameters_logit_scale_ = (
            None
        )
        logit_scale_21 = clamp_21.exp()
        clamp_21 = None
        attn_136 = attn_135 * logit_scale_21
        attn_135 = logit_scale_21 = None
        attn_137 = attn_136 + relative_position_bias_87
        attn_136 = relative_position_bias_87 = None
        attn_mask_50 = x_262.new_zeros((16, 16))
        x_262 = None
        attn_mask_50[(slice(0, -8, None), slice(0, -8, None))] = 0
        setitem_90 = attn_mask_50
        setitem_90 = None
        attn_mask_50[(slice(0, -8, None), slice(-8, -4, None))] = 1
        setitem_91 = attn_mask_50
        setitem_91 = None
        attn_mask_50[(slice(0, -8, None), slice(-4, None, None))] = 2
        setitem_92 = attn_mask_50
        setitem_92 = None
        attn_mask_50[(slice(-8, -4, None), slice(0, -8, None))] = 3
        setitem_93 = attn_mask_50
        setitem_93 = None
        attn_mask_50[(slice(-8, -4, None), slice(-8, -4, None))] = 4
        setitem_94 = attn_mask_50
        setitem_94 = None
        attn_mask_50[(slice(-8, -4, None), slice(-4, None, None))] = 5
        setitem_95 = attn_mask_50
        setitem_95 = None
        attn_mask_50[(slice(-4, None, None), slice(0, -8, None))] = 6
        setitem_96 = attn_mask_50
        setitem_96 = None
        attn_mask_50[(slice(-4, None, None), slice(-8, -4, None))] = 7
        setitem_97 = attn_mask_50
        setitem_97 = None
        attn_mask_50[(slice(-4, None, None), slice(-4, None, None))] = 8
        setitem_98 = attn_mask_50
        setitem_98 = None
        attn_mask_51 = attn_mask_50.view(2, 8, 2, 8)
        attn_mask_50 = None
        permute_98 = attn_mask_51.permute(0, 2, 1, 3)
        attn_mask_51 = None
        attn_mask_52 = permute_98.reshape(4, 64)
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
        attn_138 = attn_137.view(1, 4, 12, 64, 64)
        attn_137 = None
        unsqueeze_64 = attn_mask_54.unsqueeze(1)
        attn_mask_54 = None
        unsqueeze_65 = unsqueeze_64.unsqueeze(0)
        unsqueeze_64 = None
        attn_139 = attn_138 + unsqueeze_65
        attn_138 = unsqueeze_65 = None
        attn_140 = attn_139.view(-1, 12, 64, 64)
        attn_139 = None
        attn_141 = torch.nn.functional.softmax(attn_140, dim=-1)
        attn_140 = None
        attn_142 = torch.nn.functional.dropout(attn_141, p=0.0, training=False)
        attn_141 = None
        matmul_43 = attn_142.matmul(v_21)
        attn_142 = v_21 = None
        transpose_43 = matmul_43.transpose(1, 2)
        matmul_43 = None
        x_263 = transpose_43.reshape(4, 64, 384)
        transpose_43 = None
        x_264 = torch._C._nn.linear(
            x_263,
            l_self_modules_features_modules_5_modules_17_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_5_modules_17_modules_attn_modules_proj_parameters_bias_,
        )
        x_263 = l_self_modules_features_modules_5_modules_17_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_5_modules_17_modules_attn_modules_proj_parameters_bias_ = (None)
        x_265 = torch.nn.functional.dropout(x_264, p=0.0, training=False)
        x_264 = None
        x_266 = x_265.view(1, 2, 2, 8, 8, 384)
        x_265 = None
        permute_99 = x_266.permute(0, 1, 3, 2, 4, 5)
        x_266 = None
        x_267 = permute_99.reshape(1, 16, 16, 384)
        permute_99 = None
        x_268 = torch.roll(x_267, shifts=(4, 4), dims=(1, 2))
        x_267 = None
        getitem_139 = x_268[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_268 = None
        x_269 = getitem_139.contiguous()
        getitem_139 = None
        layer_norm_45 = torch.nn.functional.layer_norm(
            x_269,
            (384,),
            l_self_modules_features_modules_5_modules_17_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_5_modules_17_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_269 = l_self_modules_features_modules_5_modules_17_modules_norm1_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_17_modules_norm1_parameters_bias_
        ) = None
        _log_api_usage_once_42 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_42 = None
        x_270 = x_258 + layer_norm_45
        x_258 = layer_norm_45 = None
        input_175 = torch._C._nn.linear(
            x_270,
            l_self_modules_features_modules_5_modules_17_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_17_modules_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_5_modules_17_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_17_modules_mlp_modules_0_parameters_bias_ = (None)
        input_176 = torch._C._nn.gelu(input_175, approximate="none")
        input_175 = None
        input_177 = torch.nn.functional.dropout(input_176, 0.0, False, False)
        input_176 = None
        input_178 = torch._C._nn.linear(
            input_177,
            l_self_modules_features_modules_5_modules_17_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_17_modules_mlp_modules_3_parameters_bias_,
        )
        input_177 = l_self_modules_features_modules_5_modules_17_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_17_modules_mlp_modules_3_parameters_bias_ = (None)
        input_179 = torch.nn.functional.dropout(input_178, 0.0, False, False)
        input_178 = None
        layer_norm_46 = torch.nn.functional.layer_norm(
            input_179,
            (384,),
            l_self_modules_features_modules_5_modules_17_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_5_modules_17_modules_norm2_parameters_bias_,
            1e-05,
        )
        input_179 = l_self_modules_features_modules_5_modules_17_modules_norm2_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_17_modules_norm2_parameters_bias_
        ) = None
        _log_api_usage_once_43 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_43 = None
        x_271 = x_270 + layer_norm_46
        x_270 = layer_norm_46 = None
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
        x_274 = torch._C._nn.linear(
            x_273,
            l_self_modules_features_modules_6_modules_reduction_parameters_weight_,
            None,
        )
        x_273 = (
            l_self_modules_features_modules_6_modules_reduction_parameters_weight_
        ) = None
        x_275 = torch.nn.functional.layer_norm(
            x_274,
            (768,),
            l_self_modules_features_modules_6_modules_norm_parameters_weight_,
            l_self_modules_features_modules_6_modules_norm_parameters_bias_,
            1e-05,
        )
        x_274 = (
            l_self_modules_features_modules_6_modules_norm_parameters_weight_
        ) = l_self_modules_features_modules_6_modules_norm_parameters_bias_ = None
        input_180 = torch._C._nn.linear(
            l_self_modules_features_modules_7_modules_0_modules_attn_buffers_relative_coords_table_,
            l_self_modules_features_modules_7_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_7_modules_0_modules_attn_buffers_relative_coords_table_ = l_self_modules_features_modules_7_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = (None)
        input_181 = torch.nn.functional.relu(input_180, inplace=True)
        input_180 = None
        input_182 = torch._C._nn.linear(
            input_181,
            l_self_modules_features_modules_7_modules_0_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_,
            None,
        )
        input_181 = l_self_modules_features_modules_7_modules_0_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = (None)
        view_121 = input_182.view(-1, 24)
        input_182 = None
        relative_position_bias_88 = view_121[
            l_self_modules_features_modules_7_modules_0_modules_attn_buffers_relative_position_index_
        ]
        view_121 = l_self_modules_features_modules_7_modules_0_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_89 = relative_position_bias_88.view(64, 64, -1)
        relative_position_bias_88 = None
        permute_100 = relative_position_bias_89.permute(2, 0, 1)
        relative_position_bias_89 = None
        contiguous_44 = permute_100.contiguous()
        permute_100 = None
        relative_position_bias_90 = contiguous_44.unsqueeze(0)
        contiguous_44 = None
        sigmoid_22 = torch.sigmoid(relative_position_bias_90)
        relative_position_bias_90 = None
        relative_position_bias_91 = 16 * sigmoid_22
        sigmoid_22 = None
        x_276 = torch._C._nn.pad(x_275, (0, 0, 0, 1, 0, 1), "constant", None)
        x_277 = x_276.view(1, 1, 8, 1, 8, 768)
        x_276 = None
        permute_101 = x_277.permute(0, 1, 3, 2, 4, 5)
        x_277 = None
        x_278 = permute_101.reshape(1, 64, 768)
        permute_101 = None
        qkv_bias_22 = (
            l_self_modules_features_modules_7_modules_0_modules_attn_modules_qkv_parameters_bias_.clone()
        )
        l_self_modules_features_modules_7_modules_0_modules_attn_modules_qkv_parameters_bias_ = (
            None
        )
        getitem_145 = qkv_bias_22[slice(768, 1536, None)]
        zero__22 = getitem_145.zero_()
        getitem_145 = zero__22 = None
        qkv_44 = torch._C._nn.linear(
            x_278,
            l_self_modules_features_modules_7_modules_0_modules_attn_modules_qkv_parameters_weight_,
            qkv_bias_22,
        )
        x_278 = l_self_modules_features_modules_7_modules_0_modules_attn_modules_qkv_parameters_weight_ = (qkv_bias_22) = (
            None
        )
        reshape_100 = qkv_44.reshape(1, 64, 3, 24, 32)
        qkv_44 = None
        qkv_45 = reshape_100.permute(2, 0, 3, 1, 4)
        reshape_100 = None
        q_22 = qkv_45[0]
        k_22 = qkv_45[1]
        v_22 = qkv_45[2]
        qkv_45 = None
        normalize_44 = torch.nn.functional.normalize(q_22, dim=-1)
        q_22 = None
        normalize_45 = torch.nn.functional.normalize(k_22, dim=-1)
        k_22 = None
        transpose_44 = normalize_45.transpose(-2, -1)
        normalize_45 = None
        attn_143 = normalize_44 @ transpose_44
        normalize_44 = transpose_44 = None
        clamp_22 = torch.clamp(
            l_self_modules_features_modules_7_modules_0_modules_attn_parameters_logit_scale_,
            max=4.605170185988092,
        )
        l_self_modules_features_modules_7_modules_0_modules_attn_parameters_logit_scale_ = (
            None
        )
        logit_scale_22 = clamp_22.exp()
        clamp_22 = None
        attn_144 = attn_143 * logit_scale_22
        attn_143 = logit_scale_22 = None
        attn_145 = attn_144 + relative_position_bias_91
        attn_144 = relative_position_bias_91 = None
        attn_146 = torch.nn.functional.softmax(attn_145, dim=-1)
        attn_145 = None
        attn_147 = torch.nn.functional.dropout(attn_146, p=0.0, training=False)
        attn_146 = None
        matmul_45 = attn_147.matmul(v_22)
        attn_147 = v_22 = None
        transpose_45 = matmul_45.transpose(1, 2)
        matmul_45 = None
        x_279 = transpose_45.reshape(1, 64, 768)
        transpose_45 = None
        x_280 = torch._C._nn.linear(
            x_279,
            l_self_modules_features_modules_7_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_279 = l_self_modules_features_modules_7_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_281 = torch.nn.functional.dropout(x_280, p=0.0, training=False)
        x_280 = None
        x_282 = x_281.view(1, 1, 1, 8, 8, 768)
        x_281 = None
        permute_103 = x_282.permute(0, 1, 3, 2, 4, 5)
        x_282 = None
        x_283 = permute_103.reshape(1, 8, 8, 768)
        permute_103 = None
        getitem_149 = x_283[
            (
                slice(None, None, None),
                slice(None, 7, None),
                slice(None, 7, None),
                slice(None, None, None),
            )
        ]
        x_283 = None
        x_284 = getitem_149.contiguous()
        getitem_149 = None
        layer_norm_48 = torch.nn.functional.layer_norm(
            x_284,
            (768,),
            l_self_modules_features_modules_7_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_284 = (
            l_self_modules_features_modules_7_modules_0_modules_norm1_parameters_weight_
        ) = (
            l_self_modules_features_modules_7_modules_0_modules_norm1_parameters_bias_
        ) = None
        _log_api_usage_once_44 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_44 = None
        x_285 = x_275 + layer_norm_48
        x_275 = layer_norm_48 = None
        input_183 = torch._C._nn.linear(
            x_285,
            l_self_modules_features_modules_7_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_7_modules_0_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_mlp_modules_0_parameters_bias_ = (None)
        input_184 = torch._C._nn.gelu(input_183, approximate="none")
        input_183 = None
        input_185 = torch.nn.functional.dropout(input_184, 0.0, False, False)
        input_184 = None
        input_186 = torch._C._nn.linear(
            input_185,
            l_self_modules_features_modules_7_modules_0_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_mlp_modules_3_parameters_bias_,
        )
        input_185 = l_self_modules_features_modules_7_modules_0_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_mlp_modules_3_parameters_bias_ = (None)
        input_187 = torch.nn.functional.dropout(input_186, 0.0, False, False)
        input_186 = None
        layer_norm_49 = torch.nn.functional.layer_norm(
            input_187,
            (768,),
            l_self_modules_features_modules_7_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        input_187 = (
            l_self_modules_features_modules_7_modules_0_modules_norm2_parameters_weight_
        ) = (
            l_self_modules_features_modules_7_modules_0_modules_norm2_parameters_bias_
        ) = None
        _log_api_usage_once_45 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_45 = None
        x_286 = x_285 + layer_norm_49
        x_285 = layer_norm_49 = None
        input_188 = torch._C._nn.linear(
            l_self_modules_features_modules_7_modules_1_modules_attn_buffers_relative_coords_table_,
            l_self_modules_features_modules_7_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_7_modules_1_modules_attn_buffers_relative_coords_table_ = l_self_modules_features_modules_7_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_ = (None)
        input_189 = torch.nn.functional.relu(input_188, inplace=True)
        input_188 = None
        input_190 = torch._C._nn.linear(
            input_189,
            l_self_modules_features_modules_7_modules_1_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_,
            None,
        )
        input_189 = l_self_modules_features_modules_7_modules_1_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_ = (None)
        view_125 = input_190.view(-1, 24)
        input_190 = None
        relative_position_bias_92 = view_125[
            l_self_modules_features_modules_7_modules_1_modules_attn_buffers_relative_position_index_
        ]
        view_125 = l_self_modules_features_modules_7_modules_1_modules_attn_buffers_relative_position_index_ = (None)
        relative_position_bias_93 = relative_position_bias_92.view(64, 64, -1)
        relative_position_bias_92 = None
        permute_104 = relative_position_bias_93.permute(2, 0, 1)
        relative_position_bias_93 = None
        contiguous_46 = permute_104.contiguous()
        permute_104 = None
        relative_position_bias_94 = contiguous_46.unsqueeze(0)
        contiguous_46 = None
        sigmoid_23 = torch.sigmoid(relative_position_bias_94)
        relative_position_bias_94 = None
        relative_position_bias_95 = 16 * sigmoid_23
        sigmoid_23 = None
        x_287 = torch._C._nn.pad(x_286, (0, 0, 0, 1, 0, 1), "constant", None)
        x_288 = x_287.view(1, 1, 8, 1, 8, 768)
        x_287 = None
        permute_105 = x_288.permute(0, 1, 3, 2, 4, 5)
        x_288 = None
        x_289 = permute_105.reshape(1, 64, 768)
        permute_105 = None
        qkv_bias_23 = (
            l_self_modules_features_modules_7_modules_1_modules_attn_modules_qkv_parameters_bias_.clone()
        )
        l_self_modules_features_modules_7_modules_1_modules_attn_modules_qkv_parameters_bias_ = (
            None
        )
        getitem_151 = qkv_bias_23[slice(768, 1536, None)]
        zero__23 = getitem_151.zero_()
        getitem_151 = zero__23 = None
        qkv_46 = torch._C._nn.linear(
            x_289,
            l_self_modules_features_modules_7_modules_1_modules_attn_modules_qkv_parameters_weight_,
            qkv_bias_23,
        )
        x_289 = l_self_modules_features_modules_7_modules_1_modules_attn_modules_qkv_parameters_weight_ = (qkv_bias_23) = (
            None
        )
        reshape_104 = qkv_46.reshape(1, 64, 3, 24, 32)
        qkv_46 = None
        qkv_47 = reshape_104.permute(2, 0, 3, 1, 4)
        reshape_104 = None
        q_23 = qkv_47[0]
        k_23 = qkv_47[1]
        v_23 = qkv_47[2]
        qkv_47 = None
        normalize_46 = torch.nn.functional.normalize(q_23, dim=-1)
        q_23 = None
        normalize_47 = torch.nn.functional.normalize(k_23, dim=-1)
        k_23 = None
        transpose_46 = normalize_47.transpose(-2, -1)
        normalize_47 = None
        attn_148 = normalize_46 @ transpose_46
        normalize_46 = transpose_46 = None
        clamp_23 = torch.clamp(
            l_self_modules_features_modules_7_modules_1_modules_attn_parameters_logit_scale_,
            max=4.605170185988092,
        )
        l_self_modules_features_modules_7_modules_1_modules_attn_parameters_logit_scale_ = (
            None
        )
        logit_scale_23 = clamp_23.exp()
        clamp_23 = None
        attn_149 = attn_148 * logit_scale_23
        attn_148 = logit_scale_23 = None
        attn_150 = attn_149 + relative_position_bias_95
        attn_149 = relative_position_bias_95 = None
        attn_151 = torch.nn.functional.softmax(attn_150, dim=-1)
        attn_150 = None
        attn_152 = torch.nn.functional.dropout(attn_151, p=0.0, training=False)
        attn_151 = None
        matmul_47 = attn_152.matmul(v_23)
        attn_152 = v_23 = None
        transpose_47 = matmul_47.transpose(1, 2)
        matmul_47 = None
        x_290 = transpose_47.reshape(1, 64, 768)
        transpose_47 = None
        x_291 = torch._C._nn.linear(
            x_290,
            l_self_modules_features_modules_7_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_290 = l_self_modules_features_modules_7_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_292 = torch.nn.functional.dropout(x_291, p=0.0, training=False)
        x_291 = None
        x_293 = x_292.view(1, 1, 1, 8, 8, 768)
        x_292 = None
        permute_107 = x_293.permute(0, 1, 3, 2, 4, 5)
        x_293 = None
        x_294 = permute_107.reshape(1, 8, 8, 768)
        permute_107 = None
        getitem_155 = x_294[
            (
                slice(None, None, None),
                slice(None, 7, None),
                slice(None, 7, None),
                slice(None, None, None),
            )
        ]
        x_294 = None
        x_295 = getitem_155.contiguous()
        getitem_155 = None
        layer_norm_50 = torch.nn.functional.layer_norm(
            x_295,
            (768,),
            l_self_modules_features_modules_7_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_295 = (
            l_self_modules_features_modules_7_modules_1_modules_norm1_parameters_weight_
        ) = (
            l_self_modules_features_modules_7_modules_1_modules_norm1_parameters_bias_
        ) = None
        _log_api_usage_once_46 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_46 = None
        x_296 = x_286 + layer_norm_50
        x_286 = layer_norm_50 = None
        input_191 = torch._C._nn.linear(
            x_296,
            l_self_modules_features_modules_7_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_features_modules_7_modules_1_modules_mlp_modules_0_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_mlp_modules_0_parameters_bias_ = (None)
        input_192 = torch._C._nn.gelu(input_191, approximate="none")
        input_191 = None
        input_193 = torch.nn.functional.dropout(input_192, 0.0, False, False)
        input_192 = None
        input_194 = torch._C._nn.linear(
            input_193,
            l_self_modules_features_modules_7_modules_1_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_mlp_modules_3_parameters_bias_,
        )
        input_193 = l_self_modules_features_modules_7_modules_1_modules_mlp_modules_3_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_mlp_modules_3_parameters_bias_ = (None)
        input_195 = torch.nn.functional.dropout(input_194, 0.0, False, False)
        input_194 = None
        layer_norm_51 = torch.nn.functional.layer_norm(
            input_195,
            (768,),
            l_self_modules_features_modules_7_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        input_195 = (
            l_self_modules_features_modules_7_modules_1_modules_norm2_parameters_weight_
        ) = (
            l_self_modules_features_modules_7_modules_1_modules_norm2_parameters_bias_
        ) = None
        _log_api_usage_once_47 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_47 = None
        x_297 = x_296 + layer_norm_51
        x_296 = layer_norm_51 = None
        x_298 = torch.nn.functional.layer_norm(
            x_297,
            (768,),
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
