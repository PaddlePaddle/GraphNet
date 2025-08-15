import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_inputs_: torch.Tensor,
        L_self_modules_backbone_modules_patch_embed_modules_projection_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embed_modules_projection_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_parameters_cls_token_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_upsample_4x_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_upsample_4x_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_upsample_4x_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_upsample_4x_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_upsample_4x_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_upsample_4x_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_upsample_4x_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_upsample_4x_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_upsample_2x_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_upsample_2x_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_bottleneck_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fpn_bottleneck_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_conv_seg_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_conv_seg_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_inputs_ = L_inputs_
        l_self_modules_backbone_modules_patch_embed_modules_projection_parameters_weight_ = L_self_modules_backbone_modules_patch_embed_modules_projection_parameters_weight_
        l_self_modules_backbone_modules_patch_embed_modules_projection_parameters_bias_ = L_self_modules_backbone_modules_patch_embed_modules_projection_parameters_bias_
        l_self_modules_backbone_parameters_cls_token_ = (
            L_self_modules_backbone_parameters_cls_token_
        )
        l_self_modules_backbone_modules_layers_modules_0_parameters_gamma_1_ = (
            L_self_modules_backbone_modules_layers_modules_0_parameters_gamma_1_
        )
        l_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_attn_parameters_v_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_attn_parameters_v_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_attn_parameters_q_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_attn_parameters_q_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_backbone_modules_layers_modules_0_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_backbone_modules_layers_modules_0_modules_attn_buffers_relative_position_index_ = L_self_modules_backbone_modules_layers_modules_0_modules_attn_buffers_relative_position_index_
        l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_0_parameters_gamma_2_ = (
            L_self_modules_backbone_modules_layers_modules_0_parameters_gamma_2_
        )
        l_self_modules_backbone_modules_layers_modules_0_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_parameters_gamma_1_ = (
            L_self_modules_backbone_modules_layers_modules_1_parameters_gamma_1_
        )
        l_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_attn_parameters_v_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_attn_parameters_v_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_attn_parameters_q_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_attn_parameters_q_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_backbone_modules_layers_modules_1_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_backbone_modules_layers_modules_1_modules_attn_buffers_relative_position_index_ = L_self_modules_backbone_modules_layers_modules_1_modules_attn_buffers_relative_position_index_
        l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_parameters_gamma_2_ = (
            L_self_modules_backbone_modules_layers_modules_1_parameters_gamma_2_
        )
        l_self_modules_backbone_modules_layers_modules_1_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_parameters_gamma_1_ = (
            L_self_modules_backbone_modules_layers_modules_2_parameters_gamma_1_
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_attn_parameters_v_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_attn_parameters_v_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_attn_parameters_q_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_attn_parameters_q_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_backbone_modules_layers_modules_2_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_backbone_modules_layers_modules_2_modules_attn_buffers_relative_position_index_ = L_self_modules_backbone_modules_layers_modules_2_modules_attn_buffers_relative_position_index_
        l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_parameters_gamma_2_ = (
            L_self_modules_backbone_modules_layers_modules_2_parameters_gamma_2_
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_3_parameters_gamma_1_ = (
            L_self_modules_backbone_modules_layers_modules_3_parameters_gamma_1_
        )
        l_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_attn_parameters_v_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_attn_parameters_v_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_attn_parameters_q_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_attn_parameters_q_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_backbone_modules_layers_modules_3_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_backbone_modules_layers_modules_3_modules_attn_buffers_relative_position_index_ = L_self_modules_backbone_modules_layers_modules_3_modules_attn_buffers_relative_position_index_
        l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_3_parameters_gamma_2_ = (
            L_self_modules_backbone_modules_layers_modules_3_parameters_gamma_2_
        )
        l_self_modules_backbone_modules_layers_modules_3_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_4_parameters_gamma_1_ = (
            L_self_modules_backbone_modules_layers_modules_4_parameters_gamma_1_
        )
        l_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_4_modules_attn_parameters_v_bias_ = L_self_modules_backbone_modules_layers_modules_4_modules_attn_parameters_v_bias_
        l_self_modules_backbone_modules_layers_modules_4_modules_attn_parameters_q_bias_ = L_self_modules_backbone_modules_layers_modules_4_modules_attn_parameters_q_bias_
        l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_4_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_backbone_modules_layers_modules_4_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_backbone_modules_layers_modules_4_modules_attn_buffers_relative_position_index_ = L_self_modules_backbone_modules_layers_modules_4_modules_attn_buffers_relative_position_index_
        l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_4_parameters_gamma_2_ = (
            L_self_modules_backbone_modules_layers_modules_4_parameters_gamma_2_
        )
        l_self_modules_backbone_modules_layers_modules_4_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_4_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_4_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_4_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_5_parameters_gamma_1_ = (
            L_self_modules_backbone_modules_layers_modules_5_parameters_gamma_1_
        )
        l_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_5_modules_attn_parameters_v_bias_ = L_self_modules_backbone_modules_layers_modules_5_modules_attn_parameters_v_bias_
        l_self_modules_backbone_modules_layers_modules_5_modules_attn_parameters_q_bias_ = L_self_modules_backbone_modules_layers_modules_5_modules_attn_parameters_q_bias_
        l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_5_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_backbone_modules_layers_modules_5_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_backbone_modules_layers_modules_5_modules_attn_buffers_relative_position_index_ = L_self_modules_backbone_modules_layers_modules_5_modules_attn_buffers_relative_position_index_
        l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_5_parameters_gamma_2_ = (
            L_self_modules_backbone_modules_layers_modules_5_parameters_gamma_2_
        )
        l_self_modules_backbone_modules_layers_modules_5_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_5_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_5_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_5_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_6_parameters_gamma_1_ = (
            L_self_modules_backbone_modules_layers_modules_6_parameters_gamma_1_
        )
        l_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_6_modules_attn_parameters_v_bias_ = L_self_modules_backbone_modules_layers_modules_6_modules_attn_parameters_v_bias_
        l_self_modules_backbone_modules_layers_modules_6_modules_attn_parameters_q_bias_ = L_self_modules_backbone_modules_layers_modules_6_modules_attn_parameters_q_bias_
        l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_6_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_backbone_modules_layers_modules_6_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_backbone_modules_layers_modules_6_modules_attn_buffers_relative_position_index_ = L_self_modules_backbone_modules_layers_modules_6_modules_attn_buffers_relative_position_index_
        l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_6_parameters_gamma_2_ = (
            L_self_modules_backbone_modules_layers_modules_6_parameters_gamma_2_
        )
        l_self_modules_backbone_modules_layers_modules_6_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_6_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_6_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_6_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_7_parameters_gamma_1_ = (
            L_self_modules_backbone_modules_layers_modules_7_parameters_gamma_1_
        )
        l_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_7_modules_attn_parameters_v_bias_ = L_self_modules_backbone_modules_layers_modules_7_modules_attn_parameters_v_bias_
        l_self_modules_backbone_modules_layers_modules_7_modules_attn_parameters_q_bias_ = L_self_modules_backbone_modules_layers_modules_7_modules_attn_parameters_q_bias_
        l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_7_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_backbone_modules_layers_modules_7_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_backbone_modules_layers_modules_7_modules_attn_buffers_relative_position_index_ = L_self_modules_backbone_modules_layers_modules_7_modules_attn_buffers_relative_position_index_
        l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_7_parameters_gamma_2_ = (
            L_self_modules_backbone_modules_layers_modules_7_parameters_gamma_2_
        )
        l_self_modules_backbone_modules_layers_modules_7_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_7_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_7_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_7_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_8_parameters_gamma_1_ = (
            L_self_modules_backbone_modules_layers_modules_8_parameters_gamma_1_
        )
        l_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_8_modules_attn_parameters_v_bias_ = L_self_modules_backbone_modules_layers_modules_8_modules_attn_parameters_v_bias_
        l_self_modules_backbone_modules_layers_modules_8_modules_attn_parameters_q_bias_ = L_self_modules_backbone_modules_layers_modules_8_modules_attn_parameters_q_bias_
        l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_8_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_backbone_modules_layers_modules_8_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_backbone_modules_layers_modules_8_modules_attn_buffers_relative_position_index_ = L_self_modules_backbone_modules_layers_modules_8_modules_attn_buffers_relative_position_index_
        l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_8_parameters_gamma_2_ = (
            L_self_modules_backbone_modules_layers_modules_8_parameters_gamma_2_
        )
        l_self_modules_backbone_modules_layers_modules_8_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_8_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_8_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_8_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_9_parameters_gamma_1_ = (
            L_self_modules_backbone_modules_layers_modules_9_parameters_gamma_1_
        )
        l_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_9_modules_attn_parameters_v_bias_ = L_self_modules_backbone_modules_layers_modules_9_modules_attn_parameters_v_bias_
        l_self_modules_backbone_modules_layers_modules_9_modules_attn_parameters_q_bias_ = L_self_modules_backbone_modules_layers_modules_9_modules_attn_parameters_q_bias_
        l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_9_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_backbone_modules_layers_modules_9_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_backbone_modules_layers_modules_9_modules_attn_buffers_relative_position_index_ = L_self_modules_backbone_modules_layers_modules_9_modules_attn_buffers_relative_position_index_
        l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_9_parameters_gamma_2_ = (
            L_self_modules_backbone_modules_layers_modules_9_parameters_gamma_2_
        )
        l_self_modules_backbone_modules_layers_modules_9_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_9_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_9_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_9_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_10_parameters_gamma_1_ = (
            L_self_modules_backbone_modules_layers_modules_10_parameters_gamma_1_
        )
        l_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_10_modules_attn_parameters_v_bias_ = L_self_modules_backbone_modules_layers_modules_10_modules_attn_parameters_v_bias_
        l_self_modules_backbone_modules_layers_modules_10_modules_attn_parameters_q_bias_ = L_self_modules_backbone_modules_layers_modules_10_modules_attn_parameters_q_bias_
        l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_10_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_backbone_modules_layers_modules_10_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_backbone_modules_layers_modules_10_modules_attn_buffers_relative_position_index_ = L_self_modules_backbone_modules_layers_modules_10_modules_attn_buffers_relative_position_index_
        l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_10_parameters_gamma_2_ = (
            L_self_modules_backbone_modules_layers_modules_10_parameters_gamma_2_
        )
        l_self_modules_backbone_modules_layers_modules_10_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_10_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_10_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_10_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_11_parameters_gamma_1_ = (
            L_self_modules_backbone_modules_layers_modules_11_parameters_gamma_1_
        )
        l_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_11_modules_attn_parameters_v_bias_ = L_self_modules_backbone_modules_layers_modules_11_modules_attn_parameters_v_bias_
        l_self_modules_backbone_modules_layers_modules_11_modules_attn_parameters_q_bias_ = L_self_modules_backbone_modules_layers_modules_11_modules_attn_parameters_q_bias_
        l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_11_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_backbone_modules_layers_modules_11_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_backbone_modules_layers_modules_11_modules_attn_buffers_relative_position_index_ = L_self_modules_backbone_modules_layers_modules_11_modules_attn_buffers_relative_position_index_
        l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_11_parameters_gamma_2_ = (
            L_self_modules_backbone_modules_layers_modules_11_parameters_gamma_2_
        )
        l_self_modules_backbone_modules_layers_modules_11_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_11_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_11_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_11_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_neck_modules_upsample_4x_modules_0_parameters_weight_ = (
            L_self_modules_neck_modules_upsample_4x_modules_0_parameters_weight_
        )
        l_self_modules_neck_modules_upsample_4x_modules_0_parameters_bias_ = (
            L_self_modules_neck_modules_upsample_4x_modules_0_parameters_bias_
        )
        l_self_modules_neck_modules_upsample_4x_modules_1_buffers_running_mean_ = (
            L_self_modules_neck_modules_upsample_4x_modules_1_buffers_running_mean_
        )
        l_self_modules_neck_modules_upsample_4x_modules_1_buffers_running_var_ = (
            L_self_modules_neck_modules_upsample_4x_modules_1_buffers_running_var_
        )
        l_self_modules_neck_modules_upsample_4x_modules_1_parameters_weight_ = (
            L_self_modules_neck_modules_upsample_4x_modules_1_parameters_weight_
        )
        l_self_modules_neck_modules_upsample_4x_modules_1_parameters_bias_ = (
            L_self_modules_neck_modules_upsample_4x_modules_1_parameters_bias_
        )
        l_self_modules_neck_modules_upsample_4x_modules_3_parameters_weight_ = (
            L_self_modules_neck_modules_upsample_4x_modules_3_parameters_weight_
        )
        l_self_modules_neck_modules_upsample_4x_modules_3_parameters_bias_ = (
            L_self_modules_neck_modules_upsample_4x_modules_3_parameters_bias_
        )
        l_self_modules_neck_modules_upsample_2x_modules_0_parameters_weight_ = (
            L_self_modules_neck_modules_upsample_2x_modules_0_parameters_weight_
        )
        l_self_modules_neck_modules_upsample_2x_modules_0_parameters_bias_ = (
            L_self_modules_neck_modules_upsample_2x_modules_0_parameters_bias_
        )
        l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_bottleneck_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_bottleneck_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_weight_ = (
            L_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_weight_
        )
        l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_bias_ = (
            L_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_bias_
        )
        l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_fpn_bottleneck_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_fpn_bottleneck_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_conv_seg_parameters_weight_ = (
            L_self_modules_decode_head_modules_conv_seg_parameters_weight_
        )
        l_self_modules_decode_head_modules_conv_seg_parameters_bias_ = (
            L_self_modules_decode_head_modules_conv_seg_parameters_bias_
        )
        x = torch.conv2d(
            l_inputs_,
            l_self_modules_backbone_modules_patch_embed_modules_projection_parameters_weight_,
            l_self_modules_backbone_modules_patch_embed_modules_projection_parameters_bias_,
            (16, 16),
            (0, 0),
            (1, 1),
            1,
        )
        l_inputs_ = l_self_modules_backbone_modules_patch_embed_modules_projection_parameters_weight_ = l_self_modules_backbone_modules_patch_embed_modules_projection_parameters_bias_ = (None)
        flatten = x.flatten(2)
        x = None
        x_1 = flatten.transpose(1, 2)
        flatten = None
        cls_tokens = l_self_modules_backbone_parameters_cls_token_.expand(1, -1, -1)
        l_self_modules_backbone_parameters_cls_token_ = None
        x_2 = torch.cat((cls_tokens, x_1), dim=1)
        cls_tokens = x_1 = None
        layer_norm = torch.nn.functional.layer_norm(
            x_2,
            (768,),
            l_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_bias_ = (None)
        k_bias = torch.zeros_like(
            l_self_modules_backbone_modules_layers_modules_0_modules_attn_parameters_v_bias_,
            requires_grad=False,
        )
        qkv_bias = torch.cat(
            (
                l_self_modules_backbone_modules_layers_modules_0_modules_attn_parameters_q_bias_,
                k_bias,
                l_self_modules_backbone_modules_layers_modules_0_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_backbone_modules_layers_modules_0_modules_attn_parameters_q_bias_ = (
            k_bias
        ) = l_self_modules_backbone_modules_layers_modules_0_modules_attn_parameters_v_bias_ = (None)
        qkv = torch._C._nn.linear(
            input=layer_norm,
            weight=l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias,
        )
        layer_norm = l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_qkv_parameters_weight_ = (qkv_bias) = (
            None
        )
        reshape = qkv.reshape(1, 1601, 3, 12, -1)
        qkv = None
        qkv_1 = reshape.permute(2, 0, 3, 1, 4)
        reshape = None
        q = qkv_1[0]
        k = qkv_1[1]
        v = qkv_1[2]
        qkv_1 = None
        q_1 = q * 0.125
        q = None
        transpose_1 = k.transpose(-2, -1)
        k = None
        attn = q_1 @ transpose_1
        q_1 = transpose_1 = None
        view = l_self_modules_backbone_modules_layers_modules_0_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_backbone_modules_layers_modules_0_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_3 = l_self_modules_backbone_modules_layers_modules_0_modules_attn_parameters_relative_position_bias_table_[
            view
        ]
        l_self_modules_backbone_modules_layers_modules_0_modules_attn_parameters_relative_position_bias_table_ = (
            view
        ) = None
        relative_position_bias = getitem_3.view(1601, 1601, -1)
        getitem_3 = None
        permute_1 = relative_position_bias.permute(2, 0, 1)
        relative_position_bias = None
        relative_position_bias_1 = permute_1.contiguous()
        permute_1 = None
        unsqueeze = relative_position_bias_1.unsqueeze(0)
        relative_position_bias_1 = None
        attn_1 = attn + unsqueeze
        attn = unsqueeze = None
        attn_2 = attn_1.softmax(dim=-1)
        attn_1 = None
        attn_3 = torch.nn.functional.dropout(attn_2, 0.0, False, False)
        attn_2 = None
        matmul_1 = attn_3 @ v
        attn_3 = v = None
        transpose_2 = matmul_1.transpose(1, 2)
        matmul_1 = None
        x_3 = transpose_2.reshape(1, 1601, 768)
        transpose_2 = None
        x_4 = torch._C._nn.linear(
            x_3,
            l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_3 = l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_5 = torch.nn.functional.dropout(x_4, 0.0, False, False)
        x_4 = None
        mul_1 = (
            l_self_modules_backbone_modules_layers_modules_0_parameters_gamma_1_ * x_5
        )
        l_self_modules_backbone_modules_layers_modules_0_parameters_gamma_1_ = (
            x_5
        ) = None
        x_6 = x_2 + mul_1
        x_2 = mul_1 = None
        layer_norm_1 = torch.nn.functional.layer_norm(
            x_6,
            (768,),
            l_self_modules_backbone_modules_layers_modules_0_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_ln2_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_0_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_ln2_parameters_bias_ = (None)
        input_1 = torch._C._nn.linear(
            layer_norm_1,
            l_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_1 = l_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_2 = torch._C._nn.gelu(input_1, approximate="none")
        input_1 = None
        input_3 = torch.nn.functional.dropout(input_2, 0.0, False, False)
        input_2 = None
        input_4 = torch._C._nn.linear(
            input_3,
            l_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_3 = l_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_5 = torch.nn.functional.dropout(input_4, 0.0, False, False)
        input_4 = None
        mul_2 = (
            l_self_modules_backbone_modules_layers_modules_0_parameters_gamma_2_
            * input_5
        )
        l_self_modules_backbone_modules_layers_modules_0_parameters_gamma_2_ = (
            input_5
        ) = None
        x_7 = x_6 + mul_2
        x_6 = mul_2 = None
        layer_norm_2 = torch.nn.functional.layer_norm(
            x_7,
            (768,),
            l_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_bias_ = (None)
        k_bias_1 = torch.zeros_like(
            l_self_modules_backbone_modules_layers_modules_1_modules_attn_parameters_v_bias_,
            requires_grad=False,
        )
        qkv_bias_1 = torch.cat(
            (
                l_self_modules_backbone_modules_layers_modules_1_modules_attn_parameters_q_bias_,
                k_bias_1,
                l_self_modules_backbone_modules_layers_modules_1_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_backbone_modules_layers_modules_1_modules_attn_parameters_q_bias_ = (
            k_bias_1
        ) = l_self_modules_backbone_modules_layers_modules_1_modules_attn_parameters_v_bias_ = (None)
        qkv_2 = torch._C._nn.linear(
            input=layer_norm_2,
            weight=l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_1,
        )
        layer_norm_2 = l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_qkv_parameters_weight_ = (qkv_bias_1) = (
            None
        )
        reshape_2 = qkv_2.reshape(1, 1601, 3, 12, -1)
        qkv_2 = None
        qkv_3 = reshape_2.permute(2, 0, 3, 1, 4)
        reshape_2 = None
        q_2 = qkv_3[0]
        k_1 = qkv_3[1]
        v_1 = qkv_3[2]
        qkv_3 = None
        q_3 = q_2 * 0.125
        q_2 = None
        transpose_3 = k_1.transpose(-2, -1)
        k_1 = None
        attn_4 = q_3 @ transpose_3
        q_3 = transpose_3 = None
        view_2 = l_self_modules_backbone_modules_layers_modules_1_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_backbone_modules_layers_modules_1_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_7 = l_self_modules_backbone_modules_layers_modules_1_modules_attn_parameters_relative_position_bias_table_[
            view_2
        ]
        l_self_modules_backbone_modules_layers_modules_1_modules_attn_parameters_relative_position_bias_table_ = (
            view_2
        ) = None
        relative_position_bias_2 = getitem_7.view(1601, 1601, -1)
        getitem_7 = None
        permute_3 = relative_position_bias_2.permute(2, 0, 1)
        relative_position_bias_2 = None
        relative_position_bias_3 = permute_3.contiguous()
        permute_3 = None
        unsqueeze_1 = relative_position_bias_3.unsqueeze(0)
        relative_position_bias_3 = None
        attn_5 = attn_4 + unsqueeze_1
        attn_4 = unsqueeze_1 = None
        attn_6 = attn_5.softmax(dim=-1)
        attn_5 = None
        attn_7 = torch.nn.functional.dropout(attn_6, 0.0, False, False)
        attn_6 = None
        matmul_3 = attn_7 @ v_1
        attn_7 = v_1 = None
        transpose_4 = matmul_3.transpose(1, 2)
        matmul_3 = None
        x_8 = transpose_4.reshape(1, 1601, 768)
        transpose_4 = None
        x_9 = torch._C._nn.linear(
            x_8,
            l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_8 = l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_10 = torch.nn.functional.dropout(x_9, 0.0, False, False)
        x_9 = None
        mul_4 = (
            l_self_modules_backbone_modules_layers_modules_1_parameters_gamma_1_ * x_10
        )
        l_self_modules_backbone_modules_layers_modules_1_parameters_gamma_1_ = (
            x_10
        ) = None
        x_11 = x_7 + mul_4
        x_7 = mul_4 = None
        layer_norm_3 = torch.nn.functional.layer_norm(
            x_11,
            (768,),
            l_self_modules_backbone_modules_layers_modules_1_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_ln2_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_1_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_ln2_parameters_bias_ = (None)
        input_6 = torch._C._nn.linear(
            layer_norm_3,
            l_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_3 = l_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_7 = torch._C._nn.gelu(input_6, approximate="none")
        input_6 = None
        input_8 = torch.nn.functional.dropout(input_7, 0.0, False, False)
        input_7 = None
        input_9 = torch._C._nn.linear(
            input_8,
            l_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_8 = l_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_10 = torch.nn.functional.dropout(input_9, 0.0, False, False)
        input_9 = None
        mul_5 = (
            l_self_modules_backbone_modules_layers_modules_1_parameters_gamma_2_
            * input_10
        )
        l_self_modules_backbone_modules_layers_modules_1_parameters_gamma_2_ = (
            input_10
        ) = None
        x_12 = x_11 + mul_5
        x_11 = mul_5 = None
        layer_norm_4 = torch.nn.functional.layer_norm(
            x_12,
            (768,),
            l_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_bias_ = (None)
        k_bias_2 = torch.zeros_like(
            l_self_modules_backbone_modules_layers_modules_2_modules_attn_parameters_v_bias_,
            requires_grad=False,
        )
        qkv_bias_2 = torch.cat(
            (
                l_self_modules_backbone_modules_layers_modules_2_modules_attn_parameters_q_bias_,
                k_bias_2,
                l_self_modules_backbone_modules_layers_modules_2_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_attn_parameters_q_bias_ = (
            k_bias_2
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_attn_parameters_v_bias_ = (None)
        qkv_4 = torch._C._nn.linear(
            input=layer_norm_4,
            weight=l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_2,
        )
        layer_norm_4 = l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_qkv_parameters_weight_ = (qkv_bias_2) = (
            None
        )
        reshape_4 = qkv_4.reshape(1, 1601, 3, 12, -1)
        qkv_4 = None
        qkv_5 = reshape_4.permute(2, 0, 3, 1, 4)
        reshape_4 = None
        q_4 = qkv_5[0]
        k_2 = qkv_5[1]
        v_2 = qkv_5[2]
        qkv_5 = None
        q_5 = q_4 * 0.125
        q_4 = None
        transpose_5 = k_2.transpose(-2, -1)
        k_2 = None
        attn_8 = q_5 @ transpose_5
        q_5 = transpose_5 = None
        view_4 = l_self_modules_backbone_modules_layers_modules_2_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_11 = l_self_modules_backbone_modules_layers_modules_2_modules_attn_parameters_relative_position_bias_table_[
            view_4
        ]
        l_self_modules_backbone_modules_layers_modules_2_modules_attn_parameters_relative_position_bias_table_ = (
            view_4
        ) = None
        relative_position_bias_4 = getitem_11.view(1601, 1601, -1)
        getitem_11 = None
        permute_5 = relative_position_bias_4.permute(2, 0, 1)
        relative_position_bias_4 = None
        relative_position_bias_5 = permute_5.contiguous()
        permute_5 = None
        unsqueeze_2 = relative_position_bias_5.unsqueeze(0)
        relative_position_bias_5 = None
        attn_9 = attn_8 + unsqueeze_2
        attn_8 = unsqueeze_2 = None
        attn_10 = attn_9.softmax(dim=-1)
        attn_9 = None
        attn_11 = torch.nn.functional.dropout(attn_10, 0.0, False, False)
        attn_10 = None
        matmul_5 = attn_11 @ v_2
        attn_11 = v_2 = None
        transpose_6 = matmul_5.transpose(1, 2)
        matmul_5 = None
        x_13 = transpose_6.reshape(1, 1601, 768)
        transpose_6 = None
        x_14 = torch._C._nn.linear(
            x_13,
            l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_proj_parameters_bias_,
        )
        x_13 = l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_proj_parameters_bias_ = (None)
        x_15 = torch.nn.functional.dropout(x_14, 0.0, False, False)
        x_14 = None
        mul_7 = (
            l_self_modules_backbone_modules_layers_modules_2_parameters_gamma_1_ * x_15
        )
        l_self_modules_backbone_modules_layers_modules_2_parameters_gamma_1_ = (
            x_15
        ) = None
        x_16 = x_12 + mul_7
        x_12 = mul_7 = None
        layer_norm_5 = torch.nn.functional.layer_norm(
            x_16,
            (768,),
            l_self_modules_backbone_modules_layers_modules_2_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_ln2_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_ln2_parameters_bias_ = (None)
        input_11 = torch._C._nn.linear(
            layer_norm_5,
            l_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_5 = l_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_12 = torch._C._nn.gelu(input_11, approximate="none")
        input_11 = None
        input_13 = torch.nn.functional.dropout(input_12, 0.0, False, False)
        input_12 = None
        input_14 = torch._C._nn.linear(
            input_13,
            l_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_13 = l_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_15 = torch.nn.functional.dropout(input_14, 0.0, False, False)
        input_14 = None
        mul_8 = (
            l_self_modules_backbone_modules_layers_modules_2_parameters_gamma_2_
            * input_15
        )
        l_self_modules_backbone_modules_layers_modules_2_parameters_gamma_2_ = (
            input_15
        ) = None
        x_17 = x_16 + mul_8
        x_16 = mul_8 = None
        layer_norm_6 = torch.nn.functional.layer_norm(
            x_17,
            (768,),
            l_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_bias_ = (None)
        k_bias_3 = torch.zeros_like(
            l_self_modules_backbone_modules_layers_modules_3_modules_attn_parameters_v_bias_,
            requires_grad=False,
        )
        qkv_bias_3 = torch.cat(
            (
                l_self_modules_backbone_modules_layers_modules_3_modules_attn_parameters_q_bias_,
                k_bias_3,
                l_self_modules_backbone_modules_layers_modules_3_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_backbone_modules_layers_modules_3_modules_attn_parameters_q_bias_ = (
            k_bias_3
        ) = l_self_modules_backbone_modules_layers_modules_3_modules_attn_parameters_v_bias_ = (None)
        qkv_6 = torch._C._nn.linear(
            input=layer_norm_6,
            weight=l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_3,
        )
        layer_norm_6 = l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_qkv_parameters_weight_ = (qkv_bias_3) = (
            None
        )
        reshape_6 = qkv_6.reshape(1, 1601, 3, 12, -1)
        qkv_6 = None
        qkv_7 = reshape_6.permute(2, 0, 3, 1, 4)
        reshape_6 = None
        q_6 = qkv_7[0]
        k_3 = qkv_7[1]
        v_3 = qkv_7[2]
        qkv_7 = None
        q_7 = q_6 * 0.125
        q_6 = None
        transpose_7 = k_3.transpose(-2, -1)
        k_3 = None
        attn_12 = q_7 @ transpose_7
        q_7 = transpose_7 = None
        view_6 = l_self_modules_backbone_modules_layers_modules_3_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_backbone_modules_layers_modules_3_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_15 = l_self_modules_backbone_modules_layers_modules_3_modules_attn_parameters_relative_position_bias_table_[
            view_6
        ]
        l_self_modules_backbone_modules_layers_modules_3_modules_attn_parameters_relative_position_bias_table_ = (
            view_6
        ) = None
        relative_position_bias_6 = getitem_15.view(1601, 1601, -1)
        getitem_15 = None
        permute_7 = relative_position_bias_6.permute(2, 0, 1)
        relative_position_bias_6 = None
        relative_position_bias_7 = permute_7.contiguous()
        permute_7 = None
        unsqueeze_3 = relative_position_bias_7.unsqueeze(0)
        relative_position_bias_7 = None
        attn_13 = attn_12 + unsqueeze_3
        attn_12 = unsqueeze_3 = None
        attn_14 = attn_13.softmax(dim=-1)
        attn_13 = None
        attn_15 = torch.nn.functional.dropout(attn_14, 0.0, False, False)
        attn_14 = None
        matmul_7 = attn_15 @ v_3
        attn_15 = v_3 = None
        transpose_8 = matmul_7.transpose(1, 2)
        matmul_7 = None
        x_18 = transpose_8.reshape(1, 1601, 768)
        transpose_8 = None
        x_19 = torch._C._nn.linear(
            x_18,
            l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_proj_parameters_bias_,
        )
        x_18 = l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_proj_parameters_bias_ = (None)
        x_20 = torch.nn.functional.dropout(x_19, 0.0, False, False)
        x_19 = None
        mul_10 = (
            l_self_modules_backbone_modules_layers_modules_3_parameters_gamma_1_ * x_20
        )
        l_self_modules_backbone_modules_layers_modules_3_parameters_gamma_1_ = (
            x_20
        ) = None
        x_21 = x_17 + mul_10
        x_17 = mul_10 = None
        layer_norm_7 = torch.nn.functional.layer_norm(
            x_21,
            (768,),
            l_self_modules_backbone_modules_layers_modules_3_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_ln2_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_3_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_ln2_parameters_bias_ = (None)
        input_16 = torch._C._nn.linear(
            layer_norm_7,
            l_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_7 = l_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_17 = torch._C._nn.gelu(input_16, approximate="none")
        input_16 = None
        input_18 = torch.nn.functional.dropout(input_17, 0.0, False, False)
        input_17 = None
        input_19 = torch._C._nn.linear(
            input_18,
            l_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_18 = l_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_20 = torch.nn.functional.dropout(input_19, 0.0, False, False)
        input_19 = None
        mul_11 = (
            l_self_modules_backbone_modules_layers_modules_3_parameters_gamma_2_
            * input_20
        )
        l_self_modules_backbone_modules_layers_modules_3_parameters_gamma_2_ = (
            input_20
        ) = None
        x_22 = x_21 + mul_11
        x_21 = mul_11 = None
        out = x_22[(slice(None, None, None), slice(1, None, None))]
        reshape_8 = out.reshape(1, 40, 40, 768)
        out = None
        permute_8 = reshape_8.permute(0, 3, 1, 2)
        reshape_8 = None
        out_1 = permute_8.contiguous()
        permute_8 = None
        layer_norm_8 = torch.nn.functional.layer_norm(
            x_22,
            (768,),
            l_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_bias_ = (None)
        k_bias_4 = torch.zeros_like(
            l_self_modules_backbone_modules_layers_modules_4_modules_attn_parameters_v_bias_,
            requires_grad=False,
        )
        qkv_bias_4 = torch.cat(
            (
                l_self_modules_backbone_modules_layers_modules_4_modules_attn_parameters_q_bias_,
                k_bias_4,
                l_self_modules_backbone_modules_layers_modules_4_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_backbone_modules_layers_modules_4_modules_attn_parameters_q_bias_ = (
            k_bias_4
        ) = l_self_modules_backbone_modules_layers_modules_4_modules_attn_parameters_v_bias_ = (None)
        qkv_8 = torch._C._nn.linear(
            input=layer_norm_8,
            weight=l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_4,
        )
        layer_norm_8 = l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_qkv_parameters_weight_ = (qkv_bias_4) = (
            None
        )
        reshape_9 = qkv_8.reshape(1, 1601, 3, 12, -1)
        qkv_8 = None
        qkv_9 = reshape_9.permute(2, 0, 3, 1, 4)
        reshape_9 = None
        q_8 = qkv_9[0]
        k_4 = qkv_9[1]
        v_4 = qkv_9[2]
        qkv_9 = None
        q_9 = q_8 * 0.125
        q_8 = None
        transpose_9 = k_4.transpose(-2, -1)
        k_4 = None
        attn_16 = q_9 @ transpose_9
        q_9 = transpose_9 = None
        view_8 = l_self_modules_backbone_modules_layers_modules_4_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_backbone_modules_layers_modules_4_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_20 = l_self_modules_backbone_modules_layers_modules_4_modules_attn_parameters_relative_position_bias_table_[
            view_8
        ]
        l_self_modules_backbone_modules_layers_modules_4_modules_attn_parameters_relative_position_bias_table_ = (
            view_8
        ) = None
        relative_position_bias_8 = getitem_20.view(1601, 1601, -1)
        getitem_20 = None
        permute_10 = relative_position_bias_8.permute(2, 0, 1)
        relative_position_bias_8 = None
        relative_position_bias_9 = permute_10.contiguous()
        permute_10 = None
        unsqueeze_4 = relative_position_bias_9.unsqueeze(0)
        relative_position_bias_9 = None
        attn_17 = attn_16 + unsqueeze_4
        attn_16 = unsqueeze_4 = None
        attn_18 = attn_17.softmax(dim=-1)
        attn_17 = None
        attn_19 = torch.nn.functional.dropout(attn_18, 0.0, False, False)
        attn_18 = None
        matmul_9 = attn_19 @ v_4
        attn_19 = v_4 = None
        transpose_10 = matmul_9.transpose(1, 2)
        matmul_9 = None
        x_23 = transpose_10.reshape(1, 1601, 768)
        transpose_10 = None
        x_24 = torch._C._nn.linear(
            x_23,
            l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_proj_parameters_bias_,
        )
        x_23 = l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_proj_parameters_bias_ = (None)
        x_25 = torch.nn.functional.dropout(x_24, 0.0, False, False)
        x_24 = None
        mul_13 = (
            l_self_modules_backbone_modules_layers_modules_4_parameters_gamma_1_ * x_25
        )
        l_self_modules_backbone_modules_layers_modules_4_parameters_gamma_1_ = (
            x_25
        ) = None
        x_26 = x_22 + mul_13
        x_22 = mul_13 = None
        layer_norm_9 = torch.nn.functional.layer_norm(
            x_26,
            (768,),
            l_self_modules_backbone_modules_layers_modules_4_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_4_modules_ln2_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_4_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_4_modules_ln2_parameters_bias_ = (None)
        input_21 = torch._C._nn.linear(
            layer_norm_9,
            l_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_9 = l_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_22 = torch._C._nn.gelu(input_21, approximate="none")
        input_21 = None
        input_23 = torch.nn.functional.dropout(input_22, 0.0, False, False)
        input_22 = None
        input_24 = torch._C._nn.linear(
            input_23,
            l_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_23 = l_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_25 = torch.nn.functional.dropout(input_24, 0.0, False, False)
        input_24 = None
        mul_14 = (
            l_self_modules_backbone_modules_layers_modules_4_parameters_gamma_2_
            * input_25
        )
        l_self_modules_backbone_modules_layers_modules_4_parameters_gamma_2_ = (
            input_25
        ) = None
        x_27 = x_26 + mul_14
        x_26 = mul_14 = None
        layer_norm_10 = torch.nn.functional.layer_norm(
            x_27,
            (768,),
            l_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_bias_ = (None)
        k_bias_5 = torch.zeros_like(
            l_self_modules_backbone_modules_layers_modules_5_modules_attn_parameters_v_bias_,
            requires_grad=False,
        )
        qkv_bias_5 = torch.cat(
            (
                l_self_modules_backbone_modules_layers_modules_5_modules_attn_parameters_q_bias_,
                k_bias_5,
                l_self_modules_backbone_modules_layers_modules_5_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_backbone_modules_layers_modules_5_modules_attn_parameters_q_bias_ = (
            k_bias_5
        ) = l_self_modules_backbone_modules_layers_modules_5_modules_attn_parameters_v_bias_ = (None)
        qkv_10 = torch._C._nn.linear(
            input=layer_norm_10,
            weight=l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_5,
        )
        layer_norm_10 = l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_qkv_parameters_weight_ = (qkv_bias_5) = (
            None
        )
        reshape_11 = qkv_10.reshape(1, 1601, 3, 12, -1)
        qkv_10 = None
        qkv_11 = reshape_11.permute(2, 0, 3, 1, 4)
        reshape_11 = None
        q_10 = qkv_11[0]
        k_5 = qkv_11[1]
        v_5 = qkv_11[2]
        qkv_11 = None
        q_11 = q_10 * 0.125
        q_10 = None
        transpose_11 = k_5.transpose(-2, -1)
        k_5 = None
        attn_20 = q_11 @ transpose_11
        q_11 = transpose_11 = None
        view_10 = l_self_modules_backbone_modules_layers_modules_5_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_backbone_modules_layers_modules_5_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_24 = l_self_modules_backbone_modules_layers_modules_5_modules_attn_parameters_relative_position_bias_table_[
            view_10
        ]
        l_self_modules_backbone_modules_layers_modules_5_modules_attn_parameters_relative_position_bias_table_ = (
            view_10
        ) = None
        relative_position_bias_10 = getitem_24.view(1601, 1601, -1)
        getitem_24 = None
        permute_12 = relative_position_bias_10.permute(2, 0, 1)
        relative_position_bias_10 = None
        relative_position_bias_11 = permute_12.contiguous()
        permute_12 = None
        unsqueeze_5 = relative_position_bias_11.unsqueeze(0)
        relative_position_bias_11 = None
        attn_21 = attn_20 + unsqueeze_5
        attn_20 = unsqueeze_5 = None
        attn_22 = attn_21.softmax(dim=-1)
        attn_21 = None
        attn_23 = torch.nn.functional.dropout(attn_22, 0.0, False, False)
        attn_22 = None
        matmul_11 = attn_23 @ v_5
        attn_23 = v_5 = None
        transpose_12 = matmul_11.transpose(1, 2)
        matmul_11 = None
        x_28 = transpose_12.reshape(1, 1601, 768)
        transpose_12 = None
        x_29 = torch._C._nn.linear(
            x_28,
            l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_proj_parameters_bias_,
        )
        x_28 = l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_proj_parameters_bias_ = (None)
        x_30 = torch.nn.functional.dropout(x_29, 0.0, False, False)
        x_29 = None
        mul_16 = (
            l_self_modules_backbone_modules_layers_modules_5_parameters_gamma_1_ * x_30
        )
        l_self_modules_backbone_modules_layers_modules_5_parameters_gamma_1_ = (
            x_30
        ) = None
        x_31 = x_27 + mul_16
        x_27 = mul_16 = None
        layer_norm_11 = torch.nn.functional.layer_norm(
            x_31,
            (768,),
            l_self_modules_backbone_modules_layers_modules_5_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_5_modules_ln2_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_5_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_5_modules_ln2_parameters_bias_ = (None)
        input_26 = torch._C._nn.linear(
            layer_norm_11,
            l_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_11 = l_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_27 = torch._C._nn.gelu(input_26, approximate="none")
        input_26 = None
        input_28 = torch.nn.functional.dropout(input_27, 0.0, False, False)
        input_27 = None
        input_29 = torch._C._nn.linear(
            input_28,
            l_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_28 = l_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_30 = torch.nn.functional.dropout(input_29, 0.0, False, False)
        input_29 = None
        mul_17 = (
            l_self_modules_backbone_modules_layers_modules_5_parameters_gamma_2_
            * input_30
        )
        l_self_modules_backbone_modules_layers_modules_5_parameters_gamma_2_ = (
            input_30
        ) = None
        x_32 = x_31 + mul_17
        x_31 = mul_17 = None
        out_2 = x_32[(slice(None, None, None), slice(1, None, None))]
        reshape_13 = out_2.reshape(1, 40, 40, 768)
        out_2 = None
        permute_13 = reshape_13.permute(0, 3, 1, 2)
        reshape_13 = None
        out_3 = permute_13.contiguous()
        permute_13 = None
        layer_norm_12 = torch.nn.functional.layer_norm(
            x_32,
            (768,),
            l_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_bias_ = (None)
        k_bias_6 = torch.zeros_like(
            l_self_modules_backbone_modules_layers_modules_6_modules_attn_parameters_v_bias_,
            requires_grad=False,
        )
        qkv_bias_6 = torch.cat(
            (
                l_self_modules_backbone_modules_layers_modules_6_modules_attn_parameters_q_bias_,
                k_bias_6,
                l_self_modules_backbone_modules_layers_modules_6_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_backbone_modules_layers_modules_6_modules_attn_parameters_q_bias_ = (
            k_bias_6
        ) = l_self_modules_backbone_modules_layers_modules_6_modules_attn_parameters_v_bias_ = (None)
        qkv_12 = torch._C._nn.linear(
            input=layer_norm_12,
            weight=l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_6,
        )
        layer_norm_12 = l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_qkv_parameters_weight_ = (qkv_bias_6) = (
            None
        )
        reshape_14 = qkv_12.reshape(1, 1601, 3, 12, -1)
        qkv_12 = None
        qkv_13 = reshape_14.permute(2, 0, 3, 1, 4)
        reshape_14 = None
        q_12 = qkv_13[0]
        k_6 = qkv_13[1]
        v_6 = qkv_13[2]
        qkv_13 = None
        q_13 = q_12 * 0.125
        q_12 = None
        transpose_13 = k_6.transpose(-2, -1)
        k_6 = None
        attn_24 = q_13 @ transpose_13
        q_13 = transpose_13 = None
        view_12 = l_self_modules_backbone_modules_layers_modules_6_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_backbone_modules_layers_modules_6_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_29 = l_self_modules_backbone_modules_layers_modules_6_modules_attn_parameters_relative_position_bias_table_[
            view_12
        ]
        l_self_modules_backbone_modules_layers_modules_6_modules_attn_parameters_relative_position_bias_table_ = (
            view_12
        ) = None
        relative_position_bias_12 = getitem_29.view(1601, 1601, -1)
        getitem_29 = None
        permute_15 = relative_position_bias_12.permute(2, 0, 1)
        relative_position_bias_12 = None
        relative_position_bias_13 = permute_15.contiguous()
        permute_15 = None
        unsqueeze_6 = relative_position_bias_13.unsqueeze(0)
        relative_position_bias_13 = None
        attn_25 = attn_24 + unsqueeze_6
        attn_24 = unsqueeze_6 = None
        attn_26 = attn_25.softmax(dim=-1)
        attn_25 = None
        attn_27 = torch.nn.functional.dropout(attn_26, 0.0, False, False)
        attn_26 = None
        matmul_13 = attn_27 @ v_6
        attn_27 = v_6 = None
        transpose_14 = matmul_13.transpose(1, 2)
        matmul_13 = None
        x_33 = transpose_14.reshape(1, 1601, 768)
        transpose_14 = None
        x_34 = torch._C._nn.linear(
            x_33,
            l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_proj_parameters_bias_,
        )
        x_33 = l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_proj_parameters_bias_ = (None)
        x_35 = torch.nn.functional.dropout(x_34, 0.0, False, False)
        x_34 = None
        mul_19 = (
            l_self_modules_backbone_modules_layers_modules_6_parameters_gamma_1_ * x_35
        )
        l_self_modules_backbone_modules_layers_modules_6_parameters_gamma_1_ = (
            x_35
        ) = None
        x_36 = x_32 + mul_19
        x_32 = mul_19 = None
        layer_norm_13 = torch.nn.functional.layer_norm(
            x_36,
            (768,),
            l_self_modules_backbone_modules_layers_modules_6_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_6_modules_ln2_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_6_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_6_modules_ln2_parameters_bias_ = (None)
        input_31 = torch._C._nn.linear(
            layer_norm_13,
            l_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_13 = l_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_32 = torch._C._nn.gelu(input_31, approximate="none")
        input_31 = None
        input_33 = torch.nn.functional.dropout(input_32, 0.0, False, False)
        input_32 = None
        input_34 = torch._C._nn.linear(
            input_33,
            l_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_33 = l_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_35 = torch.nn.functional.dropout(input_34, 0.0, False, False)
        input_34 = None
        mul_20 = (
            l_self_modules_backbone_modules_layers_modules_6_parameters_gamma_2_
            * input_35
        )
        l_self_modules_backbone_modules_layers_modules_6_parameters_gamma_2_ = (
            input_35
        ) = None
        x_37 = x_36 + mul_20
        x_36 = mul_20 = None
        layer_norm_14 = torch.nn.functional.layer_norm(
            x_37,
            (768,),
            l_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_bias_ = (None)
        k_bias_7 = torch.zeros_like(
            l_self_modules_backbone_modules_layers_modules_7_modules_attn_parameters_v_bias_,
            requires_grad=False,
        )
        qkv_bias_7 = torch.cat(
            (
                l_self_modules_backbone_modules_layers_modules_7_modules_attn_parameters_q_bias_,
                k_bias_7,
                l_self_modules_backbone_modules_layers_modules_7_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_backbone_modules_layers_modules_7_modules_attn_parameters_q_bias_ = (
            k_bias_7
        ) = l_self_modules_backbone_modules_layers_modules_7_modules_attn_parameters_v_bias_ = (None)
        qkv_14 = torch._C._nn.linear(
            input=layer_norm_14,
            weight=l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_7,
        )
        layer_norm_14 = l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_qkv_parameters_weight_ = (qkv_bias_7) = (
            None
        )
        reshape_16 = qkv_14.reshape(1, 1601, 3, 12, -1)
        qkv_14 = None
        qkv_15 = reshape_16.permute(2, 0, 3, 1, 4)
        reshape_16 = None
        q_14 = qkv_15[0]
        k_7 = qkv_15[1]
        v_7 = qkv_15[2]
        qkv_15 = None
        q_15 = q_14 * 0.125
        q_14 = None
        transpose_15 = k_7.transpose(-2, -1)
        k_7 = None
        attn_28 = q_15 @ transpose_15
        q_15 = transpose_15 = None
        view_14 = l_self_modules_backbone_modules_layers_modules_7_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_backbone_modules_layers_modules_7_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_33 = l_self_modules_backbone_modules_layers_modules_7_modules_attn_parameters_relative_position_bias_table_[
            view_14
        ]
        l_self_modules_backbone_modules_layers_modules_7_modules_attn_parameters_relative_position_bias_table_ = (
            view_14
        ) = None
        relative_position_bias_14 = getitem_33.view(1601, 1601, -1)
        getitem_33 = None
        permute_17 = relative_position_bias_14.permute(2, 0, 1)
        relative_position_bias_14 = None
        relative_position_bias_15 = permute_17.contiguous()
        permute_17 = None
        unsqueeze_7 = relative_position_bias_15.unsqueeze(0)
        relative_position_bias_15 = None
        attn_29 = attn_28 + unsqueeze_7
        attn_28 = unsqueeze_7 = None
        attn_30 = attn_29.softmax(dim=-1)
        attn_29 = None
        attn_31 = torch.nn.functional.dropout(attn_30, 0.0, False, False)
        attn_30 = None
        matmul_15 = attn_31 @ v_7
        attn_31 = v_7 = None
        transpose_16 = matmul_15.transpose(1, 2)
        matmul_15 = None
        x_38 = transpose_16.reshape(1, 1601, 768)
        transpose_16 = None
        x_39 = torch._C._nn.linear(
            x_38,
            l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_proj_parameters_bias_,
        )
        x_38 = l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_proj_parameters_bias_ = (None)
        x_40 = torch.nn.functional.dropout(x_39, 0.0, False, False)
        x_39 = None
        mul_22 = (
            l_self_modules_backbone_modules_layers_modules_7_parameters_gamma_1_ * x_40
        )
        l_self_modules_backbone_modules_layers_modules_7_parameters_gamma_1_ = (
            x_40
        ) = None
        x_41 = x_37 + mul_22
        x_37 = mul_22 = None
        layer_norm_15 = torch.nn.functional.layer_norm(
            x_41,
            (768,),
            l_self_modules_backbone_modules_layers_modules_7_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_7_modules_ln2_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_7_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_7_modules_ln2_parameters_bias_ = (None)
        input_36 = torch._C._nn.linear(
            layer_norm_15,
            l_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_15 = l_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_37 = torch._C._nn.gelu(input_36, approximate="none")
        input_36 = None
        input_38 = torch.nn.functional.dropout(input_37, 0.0, False, False)
        input_37 = None
        input_39 = torch._C._nn.linear(
            input_38,
            l_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_38 = l_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_40 = torch.nn.functional.dropout(input_39, 0.0, False, False)
        input_39 = None
        mul_23 = (
            l_self_modules_backbone_modules_layers_modules_7_parameters_gamma_2_
            * input_40
        )
        l_self_modules_backbone_modules_layers_modules_7_parameters_gamma_2_ = (
            input_40
        ) = None
        x_42 = x_41 + mul_23
        x_41 = mul_23 = None
        out_4 = x_42[(slice(None, None, None), slice(1, None, None))]
        reshape_18 = out_4.reshape(1, 40, 40, 768)
        out_4 = None
        permute_18 = reshape_18.permute(0, 3, 1, 2)
        reshape_18 = None
        out_5 = permute_18.contiguous()
        permute_18 = None
        layer_norm_16 = torch.nn.functional.layer_norm(
            x_42,
            (768,),
            l_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_bias_ = (None)
        k_bias_8 = torch.zeros_like(
            l_self_modules_backbone_modules_layers_modules_8_modules_attn_parameters_v_bias_,
            requires_grad=False,
        )
        qkv_bias_8 = torch.cat(
            (
                l_self_modules_backbone_modules_layers_modules_8_modules_attn_parameters_q_bias_,
                k_bias_8,
                l_self_modules_backbone_modules_layers_modules_8_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_backbone_modules_layers_modules_8_modules_attn_parameters_q_bias_ = (
            k_bias_8
        ) = l_self_modules_backbone_modules_layers_modules_8_modules_attn_parameters_v_bias_ = (None)
        qkv_16 = torch._C._nn.linear(
            input=layer_norm_16,
            weight=l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_8,
        )
        layer_norm_16 = l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_qkv_parameters_weight_ = (qkv_bias_8) = (
            None
        )
        reshape_19 = qkv_16.reshape(1, 1601, 3, 12, -1)
        qkv_16 = None
        qkv_17 = reshape_19.permute(2, 0, 3, 1, 4)
        reshape_19 = None
        q_16 = qkv_17[0]
        k_8 = qkv_17[1]
        v_8 = qkv_17[2]
        qkv_17 = None
        q_17 = q_16 * 0.125
        q_16 = None
        transpose_17 = k_8.transpose(-2, -1)
        k_8 = None
        attn_32 = q_17 @ transpose_17
        q_17 = transpose_17 = None
        view_16 = l_self_modules_backbone_modules_layers_modules_8_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_backbone_modules_layers_modules_8_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_38 = l_self_modules_backbone_modules_layers_modules_8_modules_attn_parameters_relative_position_bias_table_[
            view_16
        ]
        l_self_modules_backbone_modules_layers_modules_8_modules_attn_parameters_relative_position_bias_table_ = (
            view_16
        ) = None
        relative_position_bias_16 = getitem_38.view(1601, 1601, -1)
        getitem_38 = None
        permute_20 = relative_position_bias_16.permute(2, 0, 1)
        relative_position_bias_16 = None
        relative_position_bias_17 = permute_20.contiguous()
        permute_20 = None
        unsqueeze_8 = relative_position_bias_17.unsqueeze(0)
        relative_position_bias_17 = None
        attn_33 = attn_32 + unsqueeze_8
        attn_32 = unsqueeze_8 = None
        attn_34 = attn_33.softmax(dim=-1)
        attn_33 = None
        attn_35 = torch.nn.functional.dropout(attn_34, 0.0, False, False)
        attn_34 = None
        matmul_17 = attn_35 @ v_8
        attn_35 = v_8 = None
        transpose_18 = matmul_17.transpose(1, 2)
        matmul_17 = None
        x_43 = transpose_18.reshape(1, 1601, 768)
        transpose_18 = None
        x_44 = torch._C._nn.linear(
            x_43,
            l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_proj_parameters_bias_,
        )
        x_43 = l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_proj_parameters_bias_ = (None)
        x_45 = torch.nn.functional.dropout(x_44, 0.0, False, False)
        x_44 = None
        mul_25 = (
            l_self_modules_backbone_modules_layers_modules_8_parameters_gamma_1_ * x_45
        )
        l_self_modules_backbone_modules_layers_modules_8_parameters_gamma_1_ = (
            x_45
        ) = None
        x_46 = x_42 + mul_25
        x_42 = mul_25 = None
        layer_norm_17 = torch.nn.functional.layer_norm(
            x_46,
            (768,),
            l_self_modules_backbone_modules_layers_modules_8_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_8_modules_ln2_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_8_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_8_modules_ln2_parameters_bias_ = (None)
        input_41 = torch._C._nn.linear(
            layer_norm_17,
            l_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_17 = l_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_42 = torch._C._nn.gelu(input_41, approximate="none")
        input_41 = None
        input_43 = torch.nn.functional.dropout(input_42, 0.0, False, False)
        input_42 = None
        input_44 = torch._C._nn.linear(
            input_43,
            l_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_43 = l_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_45 = torch.nn.functional.dropout(input_44, 0.0, False, False)
        input_44 = None
        mul_26 = (
            l_self_modules_backbone_modules_layers_modules_8_parameters_gamma_2_
            * input_45
        )
        l_self_modules_backbone_modules_layers_modules_8_parameters_gamma_2_ = (
            input_45
        ) = None
        x_47 = x_46 + mul_26
        x_46 = mul_26 = None
        layer_norm_18 = torch.nn.functional.layer_norm(
            x_47,
            (768,),
            l_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_bias_ = (None)
        k_bias_9 = torch.zeros_like(
            l_self_modules_backbone_modules_layers_modules_9_modules_attn_parameters_v_bias_,
            requires_grad=False,
        )
        qkv_bias_9 = torch.cat(
            (
                l_self_modules_backbone_modules_layers_modules_9_modules_attn_parameters_q_bias_,
                k_bias_9,
                l_self_modules_backbone_modules_layers_modules_9_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_backbone_modules_layers_modules_9_modules_attn_parameters_q_bias_ = (
            k_bias_9
        ) = l_self_modules_backbone_modules_layers_modules_9_modules_attn_parameters_v_bias_ = (None)
        qkv_18 = torch._C._nn.linear(
            input=layer_norm_18,
            weight=l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_9,
        )
        layer_norm_18 = l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_qkv_parameters_weight_ = (qkv_bias_9) = (
            None
        )
        reshape_21 = qkv_18.reshape(1, 1601, 3, 12, -1)
        qkv_18 = None
        qkv_19 = reshape_21.permute(2, 0, 3, 1, 4)
        reshape_21 = None
        q_18 = qkv_19[0]
        k_9 = qkv_19[1]
        v_9 = qkv_19[2]
        qkv_19 = None
        q_19 = q_18 * 0.125
        q_18 = None
        transpose_19 = k_9.transpose(-2, -1)
        k_9 = None
        attn_36 = q_19 @ transpose_19
        q_19 = transpose_19 = None
        view_18 = l_self_modules_backbone_modules_layers_modules_9_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_backbone_modules_layers_modules_9_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_42 = l_self_modules_backbone_modules_layers_modules_9_modules_attn_parameters_relative_position_bias_table_[
            view_18
        ]
        l_self_modules_backbone_modules_layers_modules_9_modules_attn_parameters_relative_position_bias_table_ = (
            view_18
        ) = None
        relative_position_bias_18 = getitem_42.view(1601, 1601, -1)
        getitem_42 = None
        permute_22 = relative_position_bias_18.permute(2, 0, 1)
        relative_position_bias_18 = None
        relative_position_bias_19 = permute_22.contiguous()
        permute_22 = None
        unsqueeze_9 = relative_position_bias_19.unsqueeze(0)
        relative_position_bias_19 = None
        attn_37 = attn_36 + unsqueeze_9
        attn_36 = unsqueeze_9 = None
        attn_38 = attn_37.softmax(dim=-1)
        attn_37 = None
        attn_39 = torch.nn.functional.dropout(attn_38, 0.0, False, False)
        attn_38 = None
        matmul_19 = attn_39 @ v_9
        attn_39 = v_9 = None
        transpose_20 = matmul_19.transpose(1, 2)
        matmul_19 = None
        x_48 = transpose_20.reshape(1, 1601, 768)
        transpose_20 = None
        x_49 = torch._C._nn.linear(
            x_48,
            l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_proj_parameters_bias_,
        )
        x_48 = l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_proj_parameters_bias_ = (None)
        x_50 = torch.nn.functional.dropout(x_49, 0.0, False, False)
        x_49 = None
        mul_28 = (
            l_self_modules_backbone_modules_layers_modules_9_parameters_gamma_1_ * x_50
        )
        l_self_modules_backbone_modules_layers_modules_9_parameters_gamma_1_ = (
            x_50
        ) = None
        x_51 = x_47 + mul_28
        x_47 = mul_28 = None
        layer_norm_19 = torch.nn.functional.layer_norm(
            x_51,
            (768,),
            l_self_modules_backbone_modules_layers_modules_9_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_9_modules_ln2_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_9_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_9_modules_ln2_parameters_bias_ = (None)
        input_46 = torch._C._nn.linear(
            layer_norm_19,
            l_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_19 = l_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_47 = torch._C._nn.gelu(input_46, approximate="none")
        input_46 = None
        input_48 = torch.nn.functional.dropout(input_47, 0.0, False, False)
        input_47 = None
        input_49 = torch._C._nn.linear(
            input_48,
            l_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_48 = l_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_50 = torch.nn.functional.dropout(input_49, 0.0, False, False)
        input_49 = None
        mul_29 = (
            l_self_modules_backbone_modules_layers_modules_9_parameters_gamma_2_
            * input_50
        )
        l_self_modules_backbone_modules_layers_modules_9_parameters_gamma_2_ = (
            input_50
        ) = None
        x_52 = x_51 + mul_29
        x_51 = mul_29 = None
        layer_norm_20 = torch.nn.functional.layer_norm(
            x_52,
            (768,),
            l_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_bias_ = (None)
        k_bias_10 = torch.zeros_like(
            l_self_modules_backbone_modules_layers_modules_10_modules_attn_parameters_v_bias_,
            requires_grad=False,
        )
        qkv_bias_10 = torch.cat(
            (
                l_self_modules_backbone_modules_layers_modules_10_modules_attn_parameters_q_bias_,
                k_bias_10,
                l_self_modules_backbone_modules_layers_modules_10_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_backbone_modules_layers_modules_10_modules_attn_parameters_q_bias_ = (
            k_bias_10
        ) = l_self_modules_backbone_modules_layers_modules_10_modules_attn_parameters_v_bias_ = (None)
        qkv_20 = torch._C._nn.linear(
            input=layer_norm_20,
            weight=l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_10,
        )
        layer_norm_20 = l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_qkv_parameters_weight_ = (qkv_bias_10) = (
            None
        )
        reshape_23 = qkv_20.reshape(1, 1601, 3, 12, -1)
        qkv_20 = None
        qkv_21 = reshape_23.permute(2, 0, 3, 1, 4)
        reshape_23 = None
        q_20 = qkv_21[0]
        k_10 = qkv_21[1]
        v_10 = qkv_21[2]
        qkv_21 = None
        q_21 = q_20 * 0.125
        q_20 = None
        transpose_21 = k_10.transpose(-2, -1)
        k_10 = None
        attn_40 = q_21 @ transpose_21
        q_21 = transpose_21 = None
        view_20 = l_self_modules_backbone_modules_layers_modules_10_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_backbone_modules_layers_modules_10_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_46 = l_self_modules_backbone_modules_layers_modules_10_modules_attn_parameters_relative_position_bias_table_[
            view_20
        ]
        l_self_modules_backbone_modules_layers_modules_10_modules_attn_parameters_relative_position_bias_table_ = (
            view_20
        ) = None
        relative_position_bias_20 = getitem_46.view(1601, 1601, -1)
        getitem_46 = None
        permute_24 = relative_position_bias_20.permute(2, 0, 1)
        relative_position_bias_20 = None
        relative_position_bias_21 = permute_24.contiguous()
        permute_24 = None
        unsqueeze_10 = relative_position_bias_21.unsqueeze(0)
        relative_position_bias_21 = None
        attn_41 = attn_40 + unsqueeze_10
        attn_40 = unsqueeze_10 = None
        attn_42 = attn_41.softmax(dim=-1)
        attn_41 = None
        attn_43 = torch.nn.functional.dropout(attn_42, 0.0, False, False)
        attn_42 = None
        matmul_21 = attn_43 @ v_10
        attn_43 = v_10 = None
        transpose_22 = matmul_21.transpose(1, 2)
        matmul_21 = None
        x_53 = transpose_22.reshape(1, 1601, 768)
        transpose_22 = None
        x_54 = torch._C._nn.linear(
            x_53,
            l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_proj_parameters_bias_,
        )
        x_53 = l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_proj_parameters_bias_ = (None)
        x_55 = torch.nn.functional.dropout(x_54, 0.0, False, False)
        x_54 = None
        mul_31 = (
            l_self_modules_backbone_modules_layers_modules_10_parameters_gamma_1_ * x_55
        )
        l_self_modules_backbone_modules_layers_modules_10_parameters_gamma_1_ = (
            x_55
        ) = None
        x_56 = x_52 + mul_31
        x_52 = mul_31 = None
        layer_norm_21 = torch.nn.functional.layer_norm(
            x_56,
            (768,),
            l_self_modules_backbone_modules_layers_modules_10_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_10_modules_ln2_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_10_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_10_modules_ln2_parameters_bias_ = (None)
        input_51 = torch._C._nn.linear(
            layer_norm_21,
            l_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_21 = l_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_52 = torch._C._nn.gelu(input_51, approximate="none")
        input_51 = None
        input_53 = torch.nn.functional.dropout(input_52, 0.0, False, False)
        input_52 = None
        input_54 = torch._C._nn.linear(
            input_53,
            l_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_53 = l_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_55 = torch.nn.functional.dropout(input_54, 0.0, False, False)
        input_54 = None
        mul_32 = (
            l_self_modules_backbone_modules_layers_modules_10_parameters_gamma_2_
            * input_55
        )
        l_self_modules_backbone_modules_layers_modules_10_parameters_gamma_2_ = (
            input_55
        ) = None
        x_57 = x_56 + mul_32
        x_56 = mul_32 = None
        layer_norm_22 = torch.nn.functional.layer_norm(
            x_57,
            (768,),
            l_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_bias_ = (None)
        k_bias_11 = torch.zeros_like(
            l_self_modules_backbone_modules_layers_modules_11_modules_attn_parameters_v_bias_,
            requires_grad=False,
        )
        qkv_bias_11 = torch.cat(
            (
                l_self_modules_backbone_modules_layers_modules_11_modules_attn_parameters_q_bias_,
                k_bias_11,
                l_self_modules_backbone_modules_layers_modules_11_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_backbone_modules_layers_modules_11_modules_attn_parameters_q_bias_ = (
            k_bias_11
        ) = l_self_modules_backbone_modules_layers_modules_11_modules_attn_parameters_v_bias_ = (None)
        qkv_22 = torch._C._nn.linear(
            input=layer_norm_22,
            weight=l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_11,
        )
        layer_norm_22 = l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_qkv_parameters_weight_ = (qkv_bias_11) = (
            None
        )
        reshape_25 = qkv_22.reshape(1, 1601, 3, 12, -1)
        qkv_22 = None
        qkv_23 = reshape_25.permute(2, 0, 3, 1, 4)
        reshape_25 = None
        q_22 = qkv_23[0]
        k_11 = qkv_23[1]
        v_11 = qkv_23[2]
        qkv_23 = None
        q_23 = q_22 * 0.125
        q_22 = None
        transpose_23 = k_11.transpose(-2, -1)
        k_11 = None
        attn_44 = q_23 @ transpose_23
        q_23 = transpose_23 = None
        view_22 = l_self_modules_backbone_modules_layers_modules_11_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_backbone_modules_layers_modules_11_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_50 = l_self_modules_backbone_modules_layers_modules_11_modules_attn_parameters_relative_position_bias_table_[
            view_22
        ]
        l_self_modules_backbone_modules_layers_modules_11_modules_attn_parameters_relative_position_bias_table_ = (
            view_22
        ) = None
        relative_position_bias_22 = getitem_50.view(1601, 1601, -1)
        getitem_50 = None
        permute_26 = relative_position_bias_22.permute(2, 0, 1)
        relative_position_bias_22 = None
        relative_position_bias_23 = permute_26.contiguous()
        permute_26 = None
        unsqueeze_11 = relative_position_bias_23.unsqueeze(0)
        relative_position_bias_23 = None
        attn_45 = attn_44 + unsqueeze_11
        attn_44 = unsqueeze_11 = None
        attn_46 = attn_45.softmax(dim=-1)
        attn_45 = None
        attn_47 = torch.nn.functional.dropout(attn_46, 0.0, False, False)
        attn_46 = None
        matmul_23 = attn_47 @ v_11
        attn_47 = v_11 = None
        transpose_24 = matmul_23.transpose(1, 2)
        matmul_23 = None
        x_58 = transpose_24.reshape(1, 1601, 768)
        transpose_24 = None
        x_59 = torch._C._nn.linear(
            x_58,
            l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_proj_parameters_bias_,
        )
        x_58 = l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_proj_parameters_bias_ = (None)
        x_60 = torch.nn.functional.dropout(x_59, 0.0, False, False)
        x_59 = None
        mul_34 = (
            l_self_modules_backbone_modules_layers_modules_11_parameters_gamma_1_ * x_60
        )
        l_self_modules_backbone_modules_layers_modules_11_parameters_gamma_1_ = (
            x_60
        ) = None
        x_61 = x_57 + mul_34
        x_57 = mul_34 = None
        layer_norm_23 = torch.nn.functional.layer_norm(
            x_61,
            (768,),
            l_self_modules_backbone_modules_layers_modules_11_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_11_modules_ln2_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_11_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_11_modules_ln2_parameters_bias_ = (None)
        input_56 = torch._C._nn.linear(
            layer_norm_23,
            l_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_23 = l_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_57 = torch._C._nn.gelu(input_56, approximate="none")
        input_56 = None
        input_58 = torch.nn.functional.dropout(input_57, 0.0, False, False)
        input_57 = None
        input_59 = torch._C._nn.linear(
            input_58,
            l_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_58 = l_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_60 = torch.nn.functional.dropout(input_59, 0.0, False, False)
        input_59 = None
        mul_35 = (
            l_self_modules_backbone_modules_layers_modules_11_parameters_gamma_2_
            * input_60
        )
        l_self_modules_backbone_modules_layers_modules_11_parameters_gamma_2_ = (
            input_60
        ) = None
        x_62 = x_61 + mul_35
        x_61 = mul_35 = None
        out_6 = x_62[(slice(None, None, None), slice(1, None, None))]
        x_62 = None
        reshape_27 = out_6.reshape(1, 40, 40, 768)
        out_6 = None
        permute_27 = reshape_27.permute(0, 3, 1, 2)
        reshape_27 = None
        out_7 = permute_27.contiguous()
        permute_27 = None
        input_61 = torch.conv_transpose2d(
            out_1,
            l_self_modules_neck_modules_upsample_4x_modules_0_parameters_weight_,
            l_self_modules_neck_modules_upsample_4x_modules_0_parameters_bias_,
            (2, 2),
            (0, 0),
            (0, 0),
            1,
            (1, 1),
        )
        out_1 = (
            l_self_modules_neck_modules_upsample_4x_modules_0_parameters_weight_
        ) = l_self_modules_neck_modules_upsample_4x_modules_0_parameters_bias_ = None
        input_62 = torch.nn.functional.batch_norm(
            input_61,
            l_self_modules_neck_modules_upsample_4x_modules_1_buffers_running_mean_,
            l_self_modules_neck_modules_upsample_4x_modules_1_buffers_running_var_,
            l_self_modules_neck_modules_upsample_4x_modules_1_parameters_weight_,
            l_self_modules_neck_modules_upsample_4x_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_61 = (
            l_self_modules_neck_modules_upsample_4x_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_neck_modules_upsample_4x_modules_1_buffers_running_var_
        ) = (
            l_self_modules_neck_modules_upsample_4x_modules_1_parameters_weight_
        ) = l_self_modules_neck_modules_upsample_4x_modules_1_parameters_bias_ = None
        input_63 = torch._C._nn.gelu(input_62, approximate="none")
        input_62 = None
        input_64 = torch.conv_transpose2d(
            input_63,
            l_self_modules_neck_modules_upsample_4x_modules_3_parameters_weight_,
            l_self_modules_neck_modules_upsample_4x_modules_3_parameters_bias_,
            (2, 2),
            (0, 0),
            (0, 0),
            1,
            (1, 1),
        )
        input_63 = (
            l_self_modules_neck_modules_upsample_4x_modules_3_parameters_weight_
        ) = l_self_modules_neck_modules_upsample_4x_modules_3_parameters_bias_ = None
        input_65 = torch.conv_transpose2d(
            out_3,
            l_self_modules_neck_modules_upsample_2x_modules_0_parameters_weight_,
            l_self_modules_neck_modules_upsample_2x_modules_0_parameters_bias_,
            (2, 2),
            (0, 0),
            (0, 0),
            1,
            (1, 1),
        )
        out_3 = (
            l_self_modules_neck_modules_upsample_2x_modules_0_parameters_weight_
        ) = l_self_modules_neck_modules_upsample_2x_modules_0_parameters_bias_ = None
        x_72 = torch.nn.functional.max_pool2d(
            out_7, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        out_7 = None
        x_63 = torch.conv2d(
            input_64,
            l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_64 = l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_conv_parameters_weight_ = (None)
        x_64 = torch.nn.functional.batch_norm(
            x_63,
            l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_63 = l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_bias_ = (None)
        x_65 = torch.nn.functional.relu(x_64, inplace=False)
        x_64 = None
        x_66 = torch.conv2d(
            input_65,
            l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_65 = l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_conv_parameters_weight_ = (None)
        x_67 = torch.nn.functional.batch_norm(
            x_66,
            l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_66 = l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_bias_ = (None)
        x_68 = torch.nn.functional.relu(x_67, inplace=False)
        x_67 = None
        x_69 = torch.conv2d(
            out_5,
            l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_5 = l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_conv_parameters_weight_ = (None)
        x_70 = torch.nn.functional.batch_norm(
            x_69,
            l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_69 = l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_bias_ = (None)
        x_71 = torch.nn.functional.relu(x_70, inplace=False)
        x_70 = None
        input_66 = torch.nn.functional.adaptive_avg_pool2d(x_72, 1)
        x_73 = torch.conv2d(
            input_66,
            l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_66 = l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_conv_parameters_weight_ = (None)
        x_74 = torch.nn.functional.batch_norm(
            x_73,
            l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_73 = l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_bias_ = (None)
        x_75 = torch.nn.functional.relu(x_74, inplace=True)
        x_74 = None
        upsampled_ppm_out = torch.nn.functional.interpolate(
            x_75, (20, 20), None, "bilinear", False
        )
        x_75 = None
        input_67 = torch.nn.functional.adaptive_avg_pool2d(x_72, 2)
        x_76 = torch.conv2d(
            input_67,
            l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_67 = l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_conv_parameters_weight_ = (None)
        x_77 = torch.nn.functional.batch_norm(
            x_76,
            l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_76 = l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_bias_ = (None)
        x_78 = torch.nn.functional.relu(x_77, inplace=True)
        x_77 = None
        upsampled_ppm_out_1 = torch.nn.functional.interpolate(
            x_78, (20, 20), None, "bilinear", False
        )
        x_78 = None
        input_68 = torch.nn.functional.adaptive_avg_pool2d(x_72, 3)
        x_79 = torch.conv2d(
            input_68,
            l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_68 = l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_conv_parameters_weight_ = (None)
        x_80 = torch.nn.functional.batch_norm(
            x_79,
            l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_79 = l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_bias_ = (None)
        x_81 = torch.nn.functional.relu(x_80, inplace=True)
        x_80 = None
        upsampled_ppm_out_2 = torch.nn.functional.interpolate(
            x_81, (20, 20), None, "bilinear", False
        )
        x_81 = None
        input_69 = torch.nn.functional.adaptive_avg_pool2d(x_72, 6)
        x_82 = torch.conv2d(
            input_69,
            l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_69 = l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_conv_parameters_weight_ = (None)
        x_83 = torch.nn.functional.batch_norm(
            x_82,
            l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_82 = l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_bias_ = (None)
        x_84 = torch.nn.functional.relu(x_83, inplace=True)
        x_83 = None
        upsampled_ppm_out_3 = torch.nn.functional.interpolate(
            x_84, (20, 20), None, "bilinear", False
        )
        x_84 = None
        psp_outs = torch.cat(
            [
                x_72,
                upsampled_ppm_out,
                upsampled_ppm_out_1,
                upsampled_ppm_out_2,
                upsampled_ppm_out_3,
            ],
            dim=1,
        )
        x_72 = (
            upsampled_ppm_out
        ) = upsampled_ppm_out_1 = upsampled_ppm_out_2 = upsampled_ppm_out_3 = None
        x_85 = torch.conv2d(
            psp_outs,
            l_self_modules_decode_head_modules_bottleneck_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        psp_outs = l_self_modules_decode_head_modules_bottleneck_modules_conv_parameters_weight_ = (None)
        x_86 = torch.nn.functional.batch_norm(
            x_85,
            l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_85 = l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_var_ = (
            l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_weight_
        ) = (
            l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_bias_
        ) = None
        x_87 = torch.nn.functional.relu(x_86, inplace=True)
        x_86 = None
        interpolate_4 = torch.nn.functional.interpolate(
            x_87, (40, 40), None, "bilinear", False
        )
        add_36 = x_71 + interpolate_4
        x_71 = interpolate_4 = None
        interpolate_5 = torch.nn.functional.interpolate(
            add_36, (80, 80), None, "bilinear", False
        )
        add_37 = x_68 + interpolate_5
        x_68 = interpolate_5 = None
        interpolate_6 = torch.nn.functional.interpolate(
            add_37, (160, 160), None, "bilinear", False
        )
        add_38 = x_65 + interpolate_6
        x_65 = interpolate_6 = None
        x_88 = torch.conv2d(
            add_38,
            l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        add_38 = l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_conv_parameters_weight_ = (None)
        x_89 = torch.nn.functional.batch_norm(
            x_88,
            l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_88 = l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_bias_ = (None)
        x_90 = torch.nn.functional.relu(x_89, inplace=False)
        x_89 = None
        x_91 = torch.conv2d(
            add_37,
            l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        add_37 = l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_conv_parameters_weight_ = (None)
        x_92 = torch.nn.functional.batch_norm(
            x_91,
            l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_91 = l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_bias_ = (None)
        x_93 = torch.nn.functional.relu(x_92, inplace=False)
        x_92 = None
        x_94 = torch.conv2d(
            add_36,
            l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        add_36 = l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_conv_parameters_weight_ = (None)
        x_95 = torch.nn.functional.batch_norm(
            x_94,
            l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_94 = l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_bias_ = (None)
        x_96 = torch.nn.functional.relu(x_95, inplace=False)
        x_95 = None
        interpolate_7 = torch.nn.functional.interpolate(
            x_87, (160, 160), None, "bilinear", False
        )
        x_87 = None
        interpolate_8 = torch.nn.functional.interpolate(
            x_96, (160, 160), None, "bilinear", False
        )
        x_96 = None
        interpolate_9 = torch.nn.functional.interpolate(
            x_93, (160, 160), None, "bilinear", False
        )
        x_93 = None
        fpn_outs = torch.cat([x_90, interpolate_9, interpolate_8, interpolate_7], dim=1)
        x_90 = interpolate_9 = interpolate_8 = interpolate_7 = None
        x_97 = torch.conv2d(
            fpn_outs,
            l_self_modules_decode_head_modules_fpn_bottleneck_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        fpn_outs = l_self_modules_decode_head_modules_fpn_bottleneck_modules_conv_parameters_weight_ = (None)
        x_98 = torch.nn.functional.batch_norm(
            x_97,
            l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_97 = l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_bias_ = (None)
        x_99 = torch.nn.functional.relu(x_98, inplace=True)
        x_98 = None
        feat = torch.nn.functional.dropout2d(x_99, 0.1, False, False)
        x_99 = None
        output = torch.conv2d(
            feat,
            l_self_modules_decode_head_modules_conv_seg_parameters_weight_,
            l_self_modules_decode_head_modules_conv_seg_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        feat = (
            l_self_modules_decode_head_modules_conv_seg_parameters_weight_
        ) = l_self_modules_decode_head_modules_conv_seg_parameters_bias_ = None
        return (output,)
