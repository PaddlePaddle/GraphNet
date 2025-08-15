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
        L_self_modules_backbone_modules_layers_modules_12_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_12_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_12_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_12_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_12_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_12_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_12_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_12_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_12_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_12_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_modules_attn_parameters_relative_position_bias_table_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_modules_attn_buffers_relative_position_index_: torch.Tensor,
        L_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_backbone_modules_layers_modules_12_parameters_gamma_1_ = (
            L_self_modules_backbone_modules_layers_modules_12_parameters_gamma_1_
        )
        l_self_modules_backbone_modules_layers_modules_12_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_12_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_12_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_12_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_12_modules_attn_parameters_v_bias_ = L_self_modules_backbone_modules_layers_modules_12_modules_attn_parameters_v_bias_
        l_self_modules_backbone_modules_layers_modules_12_modules_attn_parameters_q_bias_ = L_self_modules_backbone_modules_layers_modules_12_modules_attn_parameters_q_bias_
        l_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_12_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_backbone_modules_layers_modules_12_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_backbone_modules_layers_modules_12_modules_attn_buffers_relative_position_index_ = L_self_modules_backbone_modules_layers_modules_12_modules_attn_buffers_relative_position_index_
        l_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_12_parameters_gamma_2_ = (
            L_self_modules_backbone_modules_layers_modules_12_parameters_gamma_2_
        )
        l_self_modules_backbone_modules_layers_modules_12_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_12_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_12_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_12_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_13_parameters_gamma_1_ = (
            L_self_modules_backbone_modules_layers_modules_13_parameters_gamma_1_
        )
        l_self_modules_backbone_modules_layers_modules_13_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_13_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_13_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_13_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_13_modules_attn_parameters_v_bias_ = L_self_modules_backbone_modules_layers_modules_13_modules_attn_parameters_v_bias_
        l_self_modules_backbone_modules_layers_modules_13_modules_attn_parameters_q_bias_ = L_self_modules_backbone_modules_layers_modules_13_modules_attn_parameters_q_bias_
        l_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_13_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_backbone_modules_layers_modules_13_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_backbone_modules_layers_modules_13_modules_attn_buffers_relative_position_index_ = L_self_modules_backbone_modules_layers_modules_13_modules_attn_buffers_relative_position_index_
        l_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_13_parameters_gamma_2_ = (
            L_self_modules_backbone_modules_layers_modules_13_parameters_gamma_2_
        )
        l_self_modules_backbone_modules_layers_modules_13_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_13_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_13_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_13_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_14_parameters_gamma_1_ = (
            L_self_modules_backbone_modules_layers_modules_14_parameters_gamma_1_
        )
        l_self_modules_backbone_modules_layers_modules_14_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_14_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_14_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_14_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_14_modules_attn_parameters_v_bias_ = L_self_modules_backbone_modules_layers_modules_14_modules_attn_parameters_v_bias_
        l_self_modules_backbone_modules_layers_modules_14_modules_attn_parameters_q_bias_ = L_self_modules_backbone_modules_layers_modules_14_modules_attn_parameters_q_bias_
        l_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_14_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_backbone_modules_layers_modules_14_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_backbone_modules_layers_modules_14_modules_attn_buffers_relative_position_index_ = L_self_modules_backbone_modules_layers_modules_14_modules_attn_buffers_relative_position_index_
        l_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_14_parameters_gamma_2_ = (
            L_self_modules_backbone_modules_layers_modules_14_parameters_gamma_2_
        )
        l_self_modules_backbone_modules_layers_modules_14_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_14_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_14_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_14_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_15_parameters_gamma_1_ = (
            L_self_modules_backbone_modules_layers_modules_15_parameters_gamma_1_
        )
        l_self_modules_backbone_modules_layers_modules_15_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_15_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_15_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_15_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_15_modules_attn_parameters_v_bias_ = L_self_modules_backbone_modules_layers_modules_15_modules_attn_parameters_v_bias_
        l_self_modules_backbone_modules_layers_modules_15_modules_attn_parameters_q_bias_ = L_self_modules_backbone_modules_layers_modules_15_modules_attn_parameters_q_bias_
        l_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_15_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_backbone_modules_layers_modules_15_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_backbone_modules_layers_modules_15_modules_attn_buffers_relative_position_index_ = L_self_modules_backbone_modules_layers_modules_15_modules_attn_buffers_relative_position_index_
        l_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_15_parameters_gamma_2_ = (
            L_self_modules_backbone_modules_layers_modules_15_parameters_gamma_2_
        )
        l_self_modules_backbone_modules_layers_modules_15_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_15_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_15_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_15_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_16_parameters_gamma_1_ = (
            L_self_modules_backbone_modules_layers_modules_16_parameters_gamma_1_
        )
        l_self_modules_backbone_modules_layers_modules_16_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_16_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_16_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_16_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_16_modules_attn_parameters_v_bias_ = L_self_modules_backbone_modules_layers_modules_16_modules_attn_parameters_v_bias_
        l_self_modules_backbone_modules_layers_modules_16_modules_attn_parameters_q_bias_ = L_self_modules_backbone_modules_layers_modules_16_modules_attn_parameters_q_bias_
        l_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_16_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_backbone_modules_layers_modules_16_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_backbone_modules_layers_modules_16_modules_attn_buffers_relative_position_index_ = L_self_modules_backbone_modules_layers_modules_16_modules_attn_buffers_relative_position_index_
        l_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_16_parameters_gamma_2_ = (
            L_self_modules_backbone_modules_layers_modules_16_parameters_gamma_2_
        )
        l_self_modules_backbone_modules_layers_modules_16_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_16_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_16_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_16_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_17_parameters_gamma_1_ = (
            L_self_modules_backbone_modules_layers_modules_17_parameters_gamma_1_
        )
        l_self_modules_backbone_modules_layers_modules_17_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_17_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_17_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_17_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_17_modules_attn_parameters_v_bias_ = L_self_modules_backbone_modules_layers_modules_17_modules_attn_parameters_v_bias_
        l_self_modules_backbone_modules_layers_modules_17_modules_attn_parameters_q_bias_ = L_self_modules_backbone_modules_layers_modules_17_modules_attn_parameters_q_bias_
        l_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_17_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_backbone_modules_layers_modules_17_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_backbone_modules_layers_modules_17_modules_attn_buffers_relative_position_index_ = L_self_modules_backbone_modules_layers_modules_17_modules_attn_buffers_relative_position_index_
        l_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_17_parameters_gamma_2_ = (
            L_self_modules_backbone_modules_layers_modules_17_parameters_gamma_2_
        )
        l_self_modules_backbone_modules_layers_modules_17_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_17_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_17_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_17_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_18_parameters_gamma_1_ = (
            L_self_modules_backbone_modules_layers_modules_18_parameters_gamma_1_
        )
        l_self_modules_backbone_modules_layers_modules_18_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_18_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_18_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_18_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_18_modules_attn_parameters_v_bias_ = L_self_modules_backbone_modules_layers_modules_18_modules_attn_parameters_v_bias_
        l_self_modules_backbone_modules_layers_modules_18_modules_attn_parameters_q_bias_ = L_self_modules_backbone_modules_layers_modules_18_modules_attn_parameters_q_bias_
        l_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_18_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_backbone_modules_layers_modules_18_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_backbone_modules_layers_modules_18_modules_attn_buffers_relative_position_index_ = L_self_modules_backbone_modules_layers_modules_18_modules_attn_buffers_relative_position_index_
        l_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_18_parameters_gamma_2_ = (
            L_self_modules_backbone_modules_layers_modules_18_parameters_gamma_2_
        )
        l_self_modules_backbone_modules_layers_modules_18_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_18_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_18_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_18_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_19_parameters_gamma_1_ = (
            L_self_modules_backbone_modules_layers_modules_19_parameters_gamma_1_
        )
        l_self_modules_backbone_modules_layers_modules_19_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_19_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_19_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_19_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_19_modules_attn_parameters_v_bias_ = L_self_modules_backbone_modules_layers_modules_19_modules_attn_parameters_v_bias_
        l_self_modules_backbone_modules_layers_modules_19_modules_attn_parameters_q_bias_ = L_self_modules_backbone_modules_layers_modules_19_modules_attn_parameters_q_bias_
        l_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_19_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_backbone_modules_layers_modules_19_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_backbone_modules_layers_modules_19_modules_attn_buffers_relative_position_index_ = L_self_modules_backbone_modules_layers_modules_19_modules_attn_buffers_relative_position_index_
        l_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_19_parameters_gamma_2_ = (
            L_self_modules_backbone_modules_layers_modules_19_parameters_gamma_2_
        )
        l_self_modules_backbone_modules_layers_modules_19_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_19_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_19_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_19_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_20_parameters_gamma_1_ = (
            L_self_modules_backbone_modules_layers_modules_20_parameters_gamma_1_
        )
        l_self_modules_backbone_modules_layers_modules_20_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_20_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_20_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_20_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_20_modules_attn_parameters_v_bias_ = L_self_modules_backbone_modules_layers_modules_20_modules_attn_parameters_v_bias_
        l_self_modules_backbone_modules_layers_modules_20_modules_attn_parameters_q_bias_ = L_self_modules_backbone_modules_layers_modules_20_modules_attn_parameters_q_bias_
        l_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_20_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_backbone_modules_layers_modules_20_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_backbone_modules_layers_modules_20_modules_attn_buffers_relative_position_index_ = L_self_modules_backbone_modules_layers_modules_20_modules_attn_buffers_relative_position_index_
        l_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_20_parameters_gamma_2_ = (
            L_self_modules_backbone_modules_layers_modules_20_parameters_gamma_2_
        )
        l_self_modules_backbone_modules_layers_modules_20_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_20_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_20_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_20_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_21_parameters_gamma_1_ = (
            L_self_modules_backbone_modules_layers_modules_21_parameters_gamma_1_
        )
        l_self_modules_backbone_modules_layers_modules_21_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_21_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_21_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_21_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_21_modules_attn_parameters_v_bias_ = L_self_modules_backbone_modules_layers_modules_21_modules_attn_parameters_v_bias_
        l_self_modules_backbone_modules_layers_modules_21_modules_attn_parameters_q_bias_ = L_self_modules_backbone_modules_layers_modules_21_modules_attn_parameters_q_bias_
        l_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_21_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_backbone_modules_layers_modules_21_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_backbone_modules_layers_modules_21_modules_attn_buffers_relative_position_index_ = L_self_modules_backbone_modules_layers_modules_21_modules_attn_buffers_relative_position_index_
        l_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_21_parameters_gamma_2_ = (
            L_self_modules_backbone_modules_layers_modules_21_parameters_gamma_2_
        )
        l_self_modules_backbone_modules_layers_modules_21_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_21_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_21_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_21_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_22_parameters_gamma_1_ = (
            L_self_modules_backbone_modules_layers_modules_22_parameters_gamma_1_
        )
        l_self_modules_backbone_modules_layers_modules_22_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_22_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_22_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_22_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_22_modules_attn_parameters_v_bias_ = L_self_modules_backbone_modules_layers_modules_22_modules_attn_parameters_v_bias_
        l_self_modules_backbone_modules_layers_modules_22_modules_attn_parameters_q_bias_ = L_self_modules_backbone_modules_layers_modules_22_modules_attn_parameters_q_bias_
        l_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_22_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_backbone_modules_layers_modules_22_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_backbone_modules_layers_modules_22_modules_attn_buffers_relative_position_index_ = L_self_modules_backbone_modules_layers_modules_22_modules_attn_buffers_relative_position_index_
        l_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_22_parameters_gamma_2_ = (
            L_self_modules_backbone_modules_layers_modules_22_parameters_gamma_2_
        )
        l_self_modules_backbone_modules_layers_modules_22_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_22_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_22_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_22_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_23_parameters_gamma_1_ = (
            L_self_modules_backbone_modules_layers_modules_23_parameters_gamma_1_
        )
        l_self_modules_backbone_modules_layers_modules_23_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_23_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_23_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_23_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_23_modules_attn_parameters_v_bias_ = L_self_modules_backbone_modules_layers_modules_23_modules_attn_parameters_v_bias_
        l_self_modules_backbone_modules_layers_modules_23_modules_attn_parameters_q_bias_ = L_self_modules_backbone_modules_layers_modules_23_modules_attn_parameters_q_bias_
        l_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_23_modules_attn_parameters_relative_position_bias_table_ = L_self_modules_backbone_modules_layers_modules_23_modules_attn_parameters_relative_position_bias_table_
        l_self_modules_backbone_modules_layers_modules_23_modules_attn_buffers_relative_position_index_ = L_self_modules_backbone_modules_layers_modules_23_modules_attn_buffers_relative_position_index_
        l_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_23_parameters_gamma_2_ = (
            L_self_modules_backbone_modules_layers_modules_23_parameters_gamma_2_
        )
        l_self_modules_backbone_modules_layers_modules_23_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_23_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_23_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_23_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_1_parameters_bias_
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
            (1024,),
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
        reshape = qkv.reshape(1, 1601, 3, 16, -1)
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
        x_3 = transpose_2.reshape(1, 1601, 1024)
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
            (1024,),
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
            (1024,),
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
        reshape_2 = qkv_2.reshape(1, 1601, 3, 16, -1)
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
        x_8 = transpose_4.reshape(1, 1601, 1024)
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
            (1024,),
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
            (1024,),
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
        reshape_4 = qkv_4.reshape(1, 1601, 3, 16, -1)
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
        x_13 = transpose_6.reshape(1, 1601, 1024)
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
            (1024,),
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
            (1024,),
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
        reshape_6 = qkv_6.reshape(1, 1601, 3, 16, -1)
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
        x_18 = transpose_8.reshape(1, 1601, 1024)
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
            (1024,),
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
        layer_norm_8 = torch.nn.functional.layer_norm(
            x_22,
            (1024,),
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
        reshape_8 = qkv_8.reshape(1, 1601, 3, 16, -1)
        qkv_8 = None
        qkv_9 = reshape_8.permute(2, 0, 3, 1, 4)
        reshape_8 = None
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
        getitem_19 = l_self_modules_backbone_modules_layers_modules_4_modules_attn_parameters_relative_position_bias_table_[
            view_8
        ]
        l_self_modules_backbone_modules_layers_modules_4_modules_attn_parameters_relative_position_bias_table_ = (
            view_8
        ) = None
        relative_position_bias_8 = getitem_19.view(1601, 1601, -1)
        getitem_19 = None
        permute_9 = relative_position_bias_8.permute(2, 0, 1)
        relative_position_bias_8 = None
        relative_position_bias_9 = permute_9.contiguous()
        permute_9 = None
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
        x_23 = transpose_10.reshape(1, 1601, 1024)
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
            (1024,),
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
            (1024,),
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
        reshape_10 = qkv_10.reshape(1, 1601, 3, 16, -1)
        qkv_10 = None
        qkv_11 = reshape_10.permute(2, 0, 3, 1, 4)
        reshape_10 = None
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
        getitem_23 = l_self_modules_backbone_modules_layers_modules_5_modules_attn_parameters_relative_position_bias_table_[
            view_10
        ]
        l_self_modules_backbone_modules_layers_modules_5_modules_attn_parameters_relative_position_bias_table_ = (
            view_10
        ) = None
        relative_position_bias_10 = getitem_23.view(1601, 1601, -1)
        getitem_23 = None
        permute_11 = relative_position_bias_10.permute(2, 0, 1)
        relative_position_bias_10 = None
        relative_position_bias_11 = permute_11.contiguous()
        permute_11 = None
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
        x_28 = transpose_12.reshape(1, 1601, 1024)
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
            (1024,),
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
        layer_norm_12 = torch.nn.functional.layer_norm(
            x_32,
            (1024,),
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
        reshape_12 = qkv_12.reshape(1, 1601, 3, 16, -1)
        qkv_12 = None
        qkv_13 = reshape_12.permute(2, 0, 3, 1, 4)
        reshape_12 = None
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
        getitem_27 = l_self_modules_backbone_modules_layers_modules_6_modules_attn_parameters_relative_position_bias_table_[
            view_12
        ]
        l_self_modules_backbone_modules_layers_modules_6_modules_attn_parameters_relative_position_bias_table_ = (
            view_12
        ) = None
        relative_position_bias_12 = getitem_27.view(1601, 1601, -1)
        getitem_27 = None
        permute_13 = relative_position_bias_12.permute(2, 0, 1)
        relative_position_bias_12 = None
        relative_position_bias_13 = permute_13.contiguous()
        permute_13 = None
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
        x_33 = transpose_14.reshape(1, 1601, 1024)
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
            (1024,),
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
            (1024,),
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
        reshape_14 = qkv_14.reshape(1, 1601, 3, 16, -1)
        qkv_14 = None
        qkv_15 = reshape_14.permute(2, 0, 3, 1, 4)
        reshape_14 = None
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
        getitem_31 = l_self_modules_backbone_modules_layers_modules_7_modules_attn_parameters_relative_position_bias_table_[
            view_14
        ]
        l_self_modules_backbone_modules_layers_modules_7_modules_attn_parameters_relative_position_bias_table_ = (
            view_14
        ) = None
        relative_position_bias_14 = getitem_31.view(1601, 1601, -1)
        getitem_31 = None
        permute_15 = relative_position_bias_14.permute(2, 0, 1)
        relative_position_bias_14 = None
        relative_position_bias_15 = permute_15.contiguous()
        permute_15 = None
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
        x_38 = transpose_16.reshape(1, 1601, 1024)
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
            (1024,),
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
        out = x_42[(slice(None, None, None), slice(1, None, None))]
        reshape_16 = out.reshape(1, 40, 40, 1024)
        out = None
        permute_16 = reshape_16.permute(0, 3, 1, 2)
        reshape_16 = None
        out_1 = permute_16.contiguous()
        permute_16 = None
        layer_norm_16 = torch.nn.functional.layer_norm(
            x_42,
            (1024,),
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
        reshape_17 = qkv_16.reshape(1, 1601, 3, 16, -1)
        qkv_16 = None
        qkv_17 = reshape_17.permute(2, 0, 3, 1, 4)
        reshape_17 = None
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
        getitem_36 = l_self_modules_backbone_modules_layers_modules_8_modules_attn_parameters_relative_position_bias_table_[
            view_16
        ]
        l_self_modules_backbone_modules_layers_modules_8_modules_attn_parameters_relative_position_bias_table_ = (
            view_16
        ) = None
        relative_position_bias_16 = getitem_36.view(1601, 1601, -1)
        getitem_36 = None
        permute_18 = relative_position_bias_16.permute(2, 0, 1)
        relative_position_bias_16 = None
        relative_position_bias_17 = permute_18.contiguous()
        permute_18 = None
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
        x_43 = transpose_18.reshape(1, 1601, 1024)
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
            (1024,),
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
            (1024,),
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
        reshape_19 = qkv_18.reshape(1, 1601, 3, 16, -1)
        qkv_18 = None
        qkv_19 = reshape_19.permute(2, 0, 3, 1, 4)
        reshape_19 = None
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
        getitem_40 = l_self_modules_backbone_modules_layers_modules_9_modules_attn_parameters_relative_position_bias_table_[
            view_18
        ]
        l_self_modules_backbone_modules_layers_modules_9_modules_attn_parameters_relative_position_bias_table_ = (
            view_18
        ) = None
        relative_position_bias_18 = getitem_40.view(1601, 1601, -1)
        getitem_40 = None
        permute_20 = relative_position_bias_18.permute(2, 0, 1)
        relative_position_bias_18 = None
        relative_position_bias_19 = permute_20.contiguous()
        permute_20 = None
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
        x_48 = transpose_20.reshape(1, 1601, 1024)
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
            (1024,),
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
            (1024,),
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
        reshape_21 = qkv_20.reshape(1, 1601, 3, 16, -1)
        qkv_20 = None
        qkv_21 = reshape_21.permute(2, 0, 3, 1, 4)
        reshape_21 = None
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
        getitem_44 = l_self_modules_backbone_modules_layers_modules_10_modules_attn_parameters_relative_position_bias_table_[
            view_20
        ]
        l_self_modules_backbone_modules_layers_modules_10_modules_attn_parameters_relative_position_bias_table_ = (
            view_20
        ) = None
        relative_position_bias_20 = getitem_44.view(1601, 1601, -1)
        getitem_44 = None
        permute_22 = relative_position_bias_20.permute(2, 0, 1)
        relative_position_bias_20 = None
        relative_position_bias_21 = permute_22.contiguous()
        permute_22 = None
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
        x_53 = transpose_22.reshape(1, 1601, 1024)
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
            (1024,),
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
            (1024,),
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
        reshape_23 = qkv_22.reshape(1, 1601, 3, 16, -1)
        qkv_22 = None
        qkv_23 = reshape_23.permute(2, 0, 3, 1, 4)
        reshape_23 = None
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
        getitem_48 = l_self_modules_backbone_modules_layers_modules_11_modules_attn_parameters_relative_position_bias_table_[
            view_22
        ]
        l_self_modules_backbone_modules_layers_modules_11_modules_attn_parameters_relative_position_bias_table_ = (
            view_22
        ) = None
        relative_position_bias_22 = getitem_48.view(1601, 1601, -1)
        getitem_48 = None
        permute_24 = relative_position_bias_22.permute(2, 0, 1)
        relative_position_bias_22 = None
        relative_position_bias_23 = permute_24.contiguous()
        permute_24 = None
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
        x_58 = transpose_24.reshape(1, 1601, 1024)
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
            (1024,),
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
        out_2 = x_62[(slice(None, None, None), slice(1, None, None))]
        reshape_25 = out_2.reshape(1, 40, 40, 1024)
        out_2 = None
        permute_25 = reshape_25.permute(0, 3, 1, 2)
        reshape_25 = None
        out_3 = permute_25.contiguous()
        permute_25 = None
        layer_norm_24 = torch.nn.functional.layer_norm(
            x_62,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_12_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_12_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_12_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_12_modules_ln1_parameters_bias_ = (None)
        k_bias_12 = torch.zeros_like(
            l_self_modules_backbone_modules_layers_modules_12_modules_attn_parameters_v_bias_,
            requires_grad=False,
        )
        qkv_bias_12 = torch.cat(
            (
                l_self_modules_backbone_modules_layers_modules_12_modules_attn_parameters_q_bias_,
                k_bias_12,
                l_self_modules_backbone_modules_layers_modules_12_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_backbone_modules_layers_modules_12_modules_attn_parameters_q_bias_ = (
            k_bias_12
        ) = l_self_modules_backbone_modules_layers_modules_12_modules_attn_parameters_v_bias_ = (None)
        qkv_24 = torch._C._nn.linear(
            input=layer_norm_24,
            weight=l_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_12,
        )
        layer_norm_24 = l_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_qkv_parameters_weight_ = (qkv_bias_12) = (
            None
        )
        reshape_26 = qkv_24.reshape(1, 1601, 3, 16, -1)
        qkv_24 = None
        qkv_25 = reshape_26.permute(2, 0, 3, 1, 4)
        reshape_26 = None
        q_24 = qkv_25[0]
        k_12 = qkv_25[1]
        v_12 = qkv_25[2]
        qkv_25 = None
        q_25 = q_24 * 0.125
        q_24 = None
        transpose_25 = k_12.transpose(-2, -1)
        k_12 = None
        attn_48 = q_25 @ transpose_25
        q_25 = transpose_25 = None
        view_24 = l_self_modules_backbone_modules_layers_modules_12_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_backbone_modules_layers_modules_12_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_53 = l_self_modules_backbone_modules_layers_modules_12_modules_attn_parameters_relative_position_bias_table_[
            view_24
        ]
        l_self_modules_backbone_modules_layers_modules_12_modules_attn_parameters_relative_position_bias_table_ = (
            view_24
        ) = None
        relative_position_bias_24 = getitem_53.view(1601, 1601, -1)
        getitem_53 = None
        permute_27 = relative_position_bias_24.permute(2, 0, 1)
        relative_position_bias_24 = None
        relative_position_bias_25 = permute_27.contiguous()
        permute_27 = None
        unsqueeze_12 = relative_position_bias_25.unsqueeze(0)
        relative_position_bias_25 = None
        attn_49 = attn_48 + unsqueeze_12
        attn_48 = unsqueeze_12 = None
        attn_50 = attn_49.softmax(dim=-1)
        attn_49 = None
        attn_51 = torch.nn.functional.dropout(attn_50, 0.0, False, False)
        attn_50 = None
        matmul_25 = attn_51 @ v_12
        attn_51 = v_12 = None
        transpose_26 = matmul_25.transpose(1, 2)
        matmul_25 = None
        x_63 = transpose_26.reshape(1, 1601, 1024)
        transpose_26 = None
        x_64 = torch._C._nn.linear(
            x_63,
            l_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_proj_parameters_bias_,
        )
        x_63 = l_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_proj_parameters_bias_ = (None)
        x_65 = torch.nn.functional.dropout(x_64, 0.0, False, False)
        x_64 = None
        mul_37 = (
            l_self_modules_backbone_modules_layers_modules_12_parameters_gamma_1_ * x_65
        )
        l_self_modules_backbone_modules_layers_modules_12_parameters_gamma_1_ = (
            x_65
        ) = None
        x_66 = x_62 + mul_37
        x_62 = mul_37 = None
        layer_norm_25 = torch.nn.functional.layer_norm(
            x_66,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_12_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_12_modules_ln2_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_12_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_12_modules_ln2_parameters_bias_ = (None)
        input_61 = torch._C._nn.linear(
            layer_norm_25,
            l_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_25 = l_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_62 = torch._C._nn.gelu(input_61, approximate="none")
        input_61 = None
        input_63 = torch.nn.functional.dropout(input_62, 0.0, False, False)
        input_62 = None
        input_64 = torch._C._nn.linear(
            input_63,
            l_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_63 = l_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_65 = torch.nn.functional.dropout(input_64, 0.0, False, False)
        input_64 = None
        mul_38 = (
            l_self_modules_backbone_modules_layers_modules_12_parameters_gamma_2_
            * input_65
        )
        l_self_modules_backbone_modules_layers_modules_12_parameters_gamma_2_ = (
            input_65
        ) = None
        x_67 = x_66 + mul_38
        x_66 = mul_38 = None
        layer_norm_26 = torch.nn.functional.layer_norm(
            x_67,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_13_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_13_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_13_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_13_modules_ln1_parameters_bias_ = (None)
        k_bias_13 = torch.zeros_like(
            l_self_modules_backbone_modules_layers_modules_13_modules_attn_parameters_v_bias_,
            requires_grad=False,
        )
        qkv_bias_13 = torch.cat(
            (
                l_self_modules_backbone_modules_layers_modules_13_modules_attn_parameters_q_bias_,
                k_bias_13,
                l_self_modules_backbone_modules_layers_modules_13_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_backbone_modules_layers_modules_13_modules_attn_parameters_q_bias_ = (
            k_bias_13
        ) = l_self_modules_backbone_modules_layers_modules_13_modules_attn_parameters_v_bias_ = (None)
        qkv_26 = torch._C._nn.linear(
            input=layer_norm_26,
            weight=l_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_13,
        )
        layer_norm_26 = l_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_qkv_parameters_weight_ = (qkv_bias_13) = (
            None
        )
        reshape_28 = qkv_26.reshape(1, 1601, 3, 16, -1)
        qkv_26 = None
        qkv_27 = reshape_28.permute(2, 0, 3, 1, 4)
        reshape_28 = None
        q_26 = qkv_27[0]
        k_13 = qkv_27[1]
        v_13 = qkv_27[2]
        qkv_27 = None
        q_27 = q_26 * 0.125
        q_26 = None
        transpose_27 = k_13.transpose(-2, -1)
        k_13 = None
        attn_52 = q_27 @ transpose_27
        q_27 = transpose_27 = None
        view_26 = l_self_modules_backbone_modules_layers_modules_13_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_backbone_modules_layers_modules_13_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_57 = l_self_modules_backbone_modules_layers_modules_13_modules_attn_parameters_relative_position_bias_table_[
            view_26
        ]
        l_self_modules_backbone_modules_layers_modules_13_modules_attn_parameters_relative_position_bias_table_ = (
            view_26
        ) = None
        relative_position_bias_26 = getitem_57.view(1601, 1601, -1)
        getitem_57 = None
        permute_29 = relative_position_bias_26.permute(2, 0, 1)
        relative_position_bias_26 = None
        relative_position_bias_27 = permute_29.contiguous()
        permute_29 = None
        unsqueeze_13 = relative_position_bias_27.unsqueeze(0)
        relative_position_bias_27 = None
        attn_53 = attn_52 + unsqueeze_13
        attn_52 = unsqueeze_13 = None
        attn_54 = attn_53.softmax(dim=-1)
        attn_53 = None
        attn_55 = torch.nn.functional.dropout(attn_54, 0.0, False, False)
        attn_54 = None
        matmul_27 = attn_55 @ v_13
        attn_55 = v_13 = None
        transpose_28 = matmul_27.transpose(1, 2)
        matmul_27 = None
        x_68 = transpose_28.reshape(1, 1601, 1024)
        transpose_28 = None
        x_69 = torch._C._nn.linear(
            x_68,
            l_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_proj_parameters_bias_,
        )
        x_68 = l_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_proj_parameters_bias_ = (None)
        x_70 = torch.nn.functional.dropout(x_69, 0.0, False, False)
        x_69 = None
        mul_40 = (
            l_self_modules_backbone_modules_layers_modules_13_parameters_gamma_1_ * x_70
        )
        l_self_modules_backbone_modules_layers_modules_13_parameters_gamma_1_ = (
            x_70
        ) = None
        x_71 = x_67 + mul_40
        x_67 = mul_40 = None
        layer_norm_27 = torch.nn.functional.layer_norm(
            x_71,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_13_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_13_modules_ln2_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_13_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_13_modules_ln2_parameters_bias_ = (None)
        input_66 = torch._C._nn.linear(
            layer_norm_27,
            l_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_27 = l_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_67 = torch._C._nn.gelu(input_66, approximate="none")
        input_66 = None
        input_68 = torch.nn.functional.dropout(input_67, 0.0, False, False)
        input_67 = None
        input_69 = torch._C._nn.linear(
            input_68,
            l_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_68 = l_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_70 = torch.nn.functional.dropout(input_69, 0.0, False, False)
        input_69 = None
        mul_41 = (
            l_self_modules_backbone_modules_layers_modules_13_parameters_gamma_2_
            * input_70
        )
        l_self_modules_backbone_modules_layers_modules_13_parameters_gamma_2_ = (
            input_70
        ) = None
        x_72 = x_71 + mul_41
        x_71 = mul_41 = None
        layer_norm_28 = torch.nn.functional.layer_norm(
            x_72,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_14_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_14_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_14_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_14_modules_ln1_parameters_bias_ = (None)
        k_bias_14 = torch.zeros_like(
            l_self_modules_backbone_modules_layers_modules_14_modules_attn_parameters_v_bias_,
            requires_grad=False,
        )
        qkv_bias_14 = torch.cat(
            (
                l_self_modules_backbone_modules_layers_modules_14_modules_attn_parameters_q_bias_,
                k_bias_14,
                l_self_modules_backbone_modules_layers_modules_14_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_backbone_modules_layers_modules_14_modules_attn_parameters_q_bias_ = (
            k_bias_14
        ) = l_self_modules_backbone_modules_layers_modules_14_modules_attn_parameters_v_bias_ = (None)
        qkv_28 = torch._C._nn.linear(
            input=layer_norm_28,
            weight=l_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_14,
        )
        layer_norm_28 = l_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_qkv_parameters_weight_ = (qkv_bias_14) = (
            None
        )
        reshape_30 = qkv_28.reshape(1, 1601, 3, 16, -1)
        qkv_28 = None
        qkv_29 = reshape_30.permute(2, 0, 3, 1, 4)
        reshape_30 = None
        q_28 = qkv_29[0]
        k_14 = qkv_29[1]
        v_14 = qkv_29[2]
        qkv_29 = None
        q_29 = q_28 * 0.125
        q_28 = None
        transpose_29 = k_14.transpose(-2, -1)
        k_14 = None
        attn_56 = q_29 @ transpose_29
        q_29 = transpose_29 = None
        view_28 = l_self_modules_backbone_modules_layers_modules_14_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_backbone_modules_layers_modules_14_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_61 = l_self_modules_backbone_modules_layers_modules_14_modules_attn_parameters_relative_position_bias_table_[
            view_28
        ]
        l_self_modules_backbone_modules_layers_modules_14_modules_attn_parameters_relative_position_bias_table_ = (
            view_28
        ) = None
        relative_position_bias_28 = getitem_61.view(1601, 1601, -1)
        getitem_61 = None
        permute_31 = relative_position_bias_28.permute(2, 0, 1)
        relative_position_bias_28 = None
        relative_position_bias_29 = permute_31.contiguous()
        permute_31 = None
        unsqueeze_14 = relative_position_bias_29.unsqueeze(0)
        relative_position_bias_29 = None
        attn_57 = attn_56 + unsqueeze_14
        attn_56 = unsqueeze_14 = None
        attn_58 = attn_57.softmax(dim=-1)
        attn_57 = None
        attn_59 = torch.nn.functional.dropout(attn_58, 0.0, False, False)
        attn_58 = None
        matmul_29 = attn_59 @ v_14
        attn_59 = v_14 = None
        transpose_30 = matmul_29.transpose(1, 2)
        matmul_29 = None
        x_73 = transpose_30.reshape(1, 1601, 1024)
        transpose_30 = None
        x_74 = torch._C._nn.linear(
            x_73,
            l_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_proj_parameters_bias_,
        )
        x_73 = l_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_proj_parameters_bias_ = (None)
        x_75 = torch.nn.functional.dropout(x_74, 0.0, False, False)
        x_74 = None
        mul_43 = (
            l_self_modules_backbone_modules_layers_modules_14_parameters_gamma_1_ * x_75
        )
        l_self_modules_backbone_modules_layers_modules_14_parameters_gamma_1_ = (
            x_75
        ) = None
        x_76 = x_72 + mul_43
        x_72 = mul_43 = None
        layer_norm_29 = torch.nn.functional.layer_norm(
            x_76,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_14_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_14_modules_ln2_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_14_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_14_modules_ln2_parameters_bias_ = (None)
        input_71 = torch._C._nn.linear(
            layer_norm_29,
            l_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_29 = l_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_72 = torch._C._nn.gelu(input_71, approximate="none")
        input_71 = None
        input_73 = torch.nn.functional.dropout(input_72, 0.0, False, False)
        input_72 = None
        input_74 = torch._C._nn.linear(
            input_73,
            l_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_73 = l_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_75 = torch.nn.functional.dropout(input_74, 0.0, False, False)
        input_74 = None
        mul_44 = (
            l_self_modules_backbone_modules_layers_modules_14_parameters_gamma_2_
            * input_75
        )
        l_self_modules_backbone_modules_layers_modules_14_parameters_gamma_2_ = (
            input_75
        ) = None
        x_77 = x_76 + mul_44
        x_76 = mul_44 = None
        layer_norm_30 = torch.nn.functional.layer_norm(
            x_77,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_15_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_15_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_15_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_15_modules_ln1_parameters_bias_ = (None)
        k_bias_15 = torch.zeros_like(
            l_self_modules_backbone_modules_layers_modules_15_modules_attn_parameters_v_bias_,
            requires_grad=False,
        )
        qkv_bias_15 = torch.cat(
            (
                l_self_modules_backbone_modules_layers_modules_15_modules_attn_parameters_q_bias_,
                k_bias_15,
                l_self_modules_backbone_modules_layers_modules_15_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_backbone_modules_layers_modules_15_modules_attn_parameters_q_bias_ = (
            k_bias_15
        ) = l_self_modules_backbone_modules_layers_modules_15_modules_attn_parameters_v_bias_ = (None)
        qkv_30 = torch._C._nn.linear(
            input=layer_norm_30,
            weight=l_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_15,
        )
        layer_norm_30 = l_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_qkv_parameters_weight_ = (qkv_bias_15) = (
            None
        )
        reshape_32 = qkv_30.reshape(1, 1601, 3, 16, -1)
        qkv_30 = None
        qkv_31 = reshape_32.permute(2, 0, 3, 1, 4)
        reshape_32 = None
        q_30 = qkv_31[0]
        k_15 = qkv_31[1]
        v_15 = qkv_31[2]
        qkv_31 = None
        q_31 = q_30 * 0.125
        q_30 = None
        transpose_31 = k_15.transpose(-2, -1)
        k_15 = None
        attn_60 = q_31 @ transpose_31
        q_31 = transpose_31 = None
        view_30 = l_self_modules_backbone_modules_layers_modules_15_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_backbone_modules_layers_modules_15_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_65 = l_self_modules_backbone_modules_layers_modules_15_modules_attn_parameters_relative_position_bias_table_[
            view_30
        ]
        l_self_modules_backbone_modules_layers_modules_15_modules_attn_parameters_relative_position_bias_table_ = (
            view_30
        ) = None
        relative_position_bias_30 = getitem_65.view(1601, 1601, -1)
        getitem_65 = None
        permute_33 = relative_position_bias_30.permute(2, 0, 1)
        relative_position_bias_30 = None
        relative_position_bias_31 = permute_33.contiguous()
        permute_33 = None
        unsqueeze_15 = relative_position_bias_31.unsqueeze(0)
        relative_position_bias_31 = None
        attn_61 = attn_60 + unsqueeze_15
        attn_60 = unsqueeze_15 = None
        attn_62 = attn_61.softmax(dim=-1)
        attn_61 = None
        attn_63 = torch.nn.functional.dropout(attn_62, 0.0, False, False)
        attn_62 = None
        matmul_31 = attn_63 @ v_15
        attn_63 = v_15 = None
        transpose_32 = matmul_31.transpose(1, 2)
        matmul_31 = None
        x_78 = transpose_32.reshape(1, 1601, 1024)
        transpose_32 = None
        x_79 = torch._C._nn.linear(
            x_78,
            l_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_proj_parameters_bias_,
        )
        x_78 = l_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_proj_parameters_bias_ = (None)
        x_80 = torch.nn.functional.dropout(x_79, 0.0, False, False)
        x_79 = None
        mul_46 = (
            l_self_modules_backbone_modules_layers_modules_15_parameters_gamma_1_ * x_80
        )
        l_self_modules_backbone_modules_layers_modules_15_parameters_gamma_1_ = (
            x_80
        ) = None
        x_81 = x_77 + mul_46
        x_77 = mul_46 = None
        layer_norm_31 = torch.nn.functional.layer_norm(
            x_81,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_15_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_15_modules_ln2_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_15_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_15_modules_ln2_parameters_bias_ = (None)
        input_76 = torch._C._nn.linear(
            layer_norm_31,
            l_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_31 = l_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_77 = torch._C._nn.gelu(input_76, approximate="none")
        input_76 = None
        input_78 = torch.nn.functional.dropout(input_77, 0.0, False, False)
        input_77 = None
        input_79 = torch._C._nn.linear(
            input_78,
            l_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_78 = l_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_80 = torch.nn.functional.dropout(input_79, 0.0, False, False)
        input_79 = None
        mul_47 = (
            l_self_modules_backbone_modules_layers_modules_15_parameters_gamma_2_
            * input_80
        )
        l_self_modules_backbone_modules_layers_modules_15_parameters_gamma_2_ = (
            input_80
        ) = None
        x_82 = x_81 + mul_47
        x_81 = mul_47 = None
        out_4 = x_82[(slice(None, None, None), slice(1, None, None))]
        reshape_34 = out_4.reshape(1, 40, 40, 1024)
        out_4 = None
        permute_34 = reshape_34.permute(0, 3, 1, 2)
        reshape_34 = None
        out_5 = permute_34.contiguous()
        permute_34 = None
        layer_norm_32 = torch.nn.functional.layer_norm(
            x_82,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_16_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_16_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_16_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_16_modules_ln1_parameters_bias_ = (None)
        k_bias_16 = torch.zeros_like(
            l_self_modules_backbone_modules_layers_modules_16_modules_attn_parameters_v_bias_,
            requires_grad=False,
        )
        qkv_bias_16 = torch.cat(
            (
                l_self_modules_backbone_modules_layers_modules_16_modules_attn_parameters_q_bias_,
                k_bias_16,
                l_self_modules_backbone_modules_layers_modules_16_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_backbone_modules_layers_modules_16_modules_attn_parameters_q_bias_ = (
            k_bias_16
        ) = l_self_modules_backbone_modules_layers_modules_16_modules_attn_parameters_v_bias_ = (None)
        qkv_32 = torch._C._nn.linear(
            input=layer_norm_32,
            weight=l_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_16,
        )
        layer_norm_32 = l_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_qkv_parameters_weight_ = (qkv_bias_16) = (
            None
        )
        reshape_35 = qkv_32.reshape(1, 1601, 3, 16, -1)
        qkv_32 = None
        qkv_33 = reshape_35.permute(2, 0, 3, 1, 4)
        reshape_35 = None
        q_32 = qkv_33[0]
        k_16 = qkv_33[1]
        v_16 = qkv_33[2]
        qkv_33 = None
        q_33 = q_32 * 0.125
        q_32 = None
        transpose_33 = k_16.transpose(-2, -1)
        k_16 = None
        attn_64 = q_33 @ transpose_33
        q_33 = transpose_33 = None
        view_32 = l_self_modules_backbone_modules_layers_modules_16_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_backbone_modules_layers_modules_16_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_70 = l_self_modules_backbone_modules_layers_modules_16_modules_attn_parameters_relative_position_bias_table_[
            view_32
        ]
        l_self_modules_backbone_modules_layers_modules_16_modules_attn_parameters_relative_position_bias_table_ = (
            view_32
        ) = None
        relative_position_bias_32 = getitem_70.view(1601, 1601, -1)
        getitem_70 = None
        permute_36 = relative_position_bias_32.permute(2, 0, 1)
        relative_position_bias_32 = None
        relative_position_bias_33 = permute_36.contiguous()
        permute_36 = None
        unsqueeze_16 = relative_position_bias_33.unsqueeze(0)
        relative_position_bias_33 = None
        attn_65 = attn_64 + unsqueeze_16
        attn_64 = unsqueeze_16 = None
        attn_66 = attn_65.softmax(dim=-1)
        attn_65 = None
        attn_67 = torch.nn.functional.dropout(attn_66, 0.0, False, False)
        attn_66 = None
        matmul_33 = attn_67 @ v_16
        attn_67 = v_16 = None
        transpose_34 = matmul_33.transpose(1, 2)
        matmul_33 = None
        x_83 = transpose_34.reshape(1, 1601, 1024)
        transpose_34 = None
        x_84 = torch._C._nn.linear(
            x_83,
            l_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_proj_parameters_bias_,
        )
        x_83 = l_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_proj_parameters_bias_ = (None)
        x_85 = torch.nn.functional.dropout(x_84, 0.0, False, False)
        x_84 = None
        mul_49 = (
            l_self_modules_backbone_modules_layers_modules_16_parameters_gamma_1_ * x_85
        )
        l_self_modules_backbone_modules_layers_modules_16_parameters_gamma_1_ = (
            x_85
        ) = None
        x_86 = x_82 + mul_49
        x_82 = mul_49 = None
        layer_norm_33 = torch.nn.functional.layer_norm(
            x_86,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_16_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_16_modules_ln2_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_16_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_16_modules_ln2_parameters_bias_ = (None)
        input_81 = torch._C._nn.linear(
            layer_norm_33,
            l_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_33 = l_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_82 = torch._C._nn.gelu(input_81, approximate="none")
        input_81 = None
        input_83 = torch.nn.functional.dropout(input_82, 0.0, False, False)
        input_82 = None
        input_84 = torch._C._nn.linear(
            input_83,
            l_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_83 = l_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_85 = torch.nn.functional.dropout(input_84, 0.0, False, False)
        input_84 = None
        mul_50 = (
            l_self_modules_backbone_modules_layers_modules_16_parameters_gamma_2_
            * input_85
        )
        l_self_modules_backbone_modules_layers_modules_16_parameters_gamma_2_ = (
            input_85
        ) = None
        x_87 = x_86 + mul_50
        x_86 = mul_50 = None
        layer_norm_34 = torch.nn.functional.layer_norm(
            x_87,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_17_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_17_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_17_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_17_modules_ln1_parameters_bias_ = (None)
        k_bias_17 = torch.zeros_like(
            l_self_modules_backbone_modules_layers_modules_17_modules_attn_parameters_v_bias_,
            requires_grad=False,
        )
        qkv_bias_17 = torch.cat(
            (
                l_self_modules_backbone_modules_layers_modules_17_modules_attn_parameters_q_bias_,
                k_bias_17,
                l_self_modules_backbone_modules_layers_modules_17_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_backbone_modules_layers_modules_17_modules_attn_parameters_q_bias_ = (
            k_bias_17
        ) = l_self_modules_backbone_modules_layers_modules_17_modules_attn_parameters_v_bias_ = (None)
        qkv_34 = torch._C._nn.linear(
            input=layer_norm_34,
            weight=l_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_17,
        )
        layer_norm_34 = l_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_qkv_parameters_weight_ = (qkv_bias_17) = (
            None
        )
        reshape_37 = qkv_34.reshape(1, 1601, 3, 16, -1)
        qkv_34 = None
        qkv_35 = reshape_37.permute(2, 0, 3, 1, 4)
        reshape_37 = None
        q_34 = qkv_35[0]
        k_17 = qkv_35[1]
        v_17 = qkv_35[2]
        qkv_35 = None
        q_35 = q_34 * 0.125
        q_34 = None
        transpose_35 = k_17.transpose(-2, -1)
        k_17 = None
        attn_68 = q_35 @ transpose_35
        q_35 = transpose_35 = None
        view_34 = l_self_modules_backbone_modules_layers_modules_17_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_backbone_modules_layers_modules_17_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_74 = l_self_modules_backbone_modules_layers_modules_17_modules_attn_parameters_relative_position_bias_table_[
            view_34
        ]
        l_self_modules_backbone_modules_layers_modules_17_modules_attn_parameters_relative_position_bias_table_ = (
            view_34
        ) = None
        relative_position_bias_34 = getitem_74.view(1601, 1601, -1)
        getitem_74 = None
        permute_38 = relative_position_bias_34.permute(2, 0, 1)
        relative_position_bias_34 = None
        relative_position_bias_35 = permute_38.contiguous()
        permute_38 = None
        unsqueeze_17 = relative_position_bias_35.unsqueeze(0)
        relative_position_bias_35 = None
        attn_69 = attn_68 + unsqueeze_17
        attn_68 = unsqueeze_17 = None
        attn_70 = attn_69.softmax(dim=-1)
        attn_69 = None
        attn_71 = torch.nn.functional.dropout(attn_70, 0.0, False, False)
        attn_70 = None
        matmul_35 = attn_71 @ v_17
        attn_71 = v_17 = None
        transpose_36 = matmul_35.transpose(1, 2)
        matmul_35 = None
        x_88 = transpose_36.reshape(1, 1601, 1024)
        transpose_36 = None
        x_89 = torch._C._nn.linear(
            x_88,
            l_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_proj_parameters_bias_,
        )
        x_88 = l_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_proj_parameters_bias_ = (None)
        x_90 = torch.nn.functional.dropout(x_89, 0.0, False, False)
        x_89 = None
        mul_52 = (
            l_self_modules_backbone_modules_layers_modules_17_parameters_gamma_1_ * x_90
        )
        l_self_modules_backbone_modules_layers_modules_17_parameters_gamma_1_ = (
            x_90
        ) = None
        x_91 = x_87 + mul_52
        x_87 = mul_52 = None
        layer_norm_35 = torch.nn.functional.layer_norm(
            x_91,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_17_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_17_modules_ln2_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_17_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_17_modules_ln2_parameters_bias_ = (None)
        input_86 = torch._C._nn.linear(
            layer_norm_35,
            l_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_35 = l_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_87 = torch._C._nn.gelu(input_86, approximate="none")
        input_86 = None
        input_88 = torch.nn.functional.dropout(input_87, 0.0, False, False)
        input_87 = None
        input_89 = torch._C._nn.linear(
            input_88,
            l_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_88 = l_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_90 = torch.nn.functional.dropout(input_89, 0.0, False, False)
        input_89 = None
        mul_53 = (
            l_self_modules_backbone_modules_layers_modules_17_parameters_gamma_2_
            * input_90
        )
        l_self_modules_backbone_modules_layers_modules_17_parameters_gamma_2_ = (
            input_90
        ) = None
        x_92 = x_91 + mul_53
        x_91 = mul_53 = None
        layer_norm_36 = torch.nn.functional.layer_norm(
            x_92,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_18_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_18_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_18_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_18_modules_ln1_parameters_bias_ = (None)
        k_bias_18 = torch.zeros_like(
            l_self_modules_backbone_modules_layers_modules_18_modules_attn_parameters_v_bias_,
            requires_grad=False,
        )
        qkv_bias_18 = torch.cat(
            (
                l_self_modules_backbone_modules_layers_modules_18_modules_attn_parameters_q_bias_,
                k_bias_18,
                l_self_modules_backbone_modules_layers_modules_18_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_backbone_modules_layers_modules_18_modules_attn_parameters_q_bias_ = (
            k_bias_18
        ) = l_self_modules_backbone_modules_layers_modules_18_modules_attn_parameters_v_bias_ = (None)
        qkv_36 = torch._C._nn.linear(
            input=layer_norm_36,
            weight=l_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_18,
        )
        layer_norm_36 = l_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_qkv_parameters_weight_ = (qkv_bias_18) = (
            None
        )
        reshape_39 = qkv_36.reshape(1, 1601, 3, 16, -1)
        qkv_36 = None
        qkv_37 = reshape_39.permute(2, 0, 3, 1, 4)
        reshape_39 = None
        q_36 = qkv_37[0]
        k_18 = qkv_37[1]
        v_18 = qkv_37[2]
        qkv_37 = None
        q_37 = q_36 * 0.125
        q_36 = None
        transpose_37 = k_18.transpose(-2, -1)
        k_18 = None
        attn_72 = q_37 @ transpose_37
        q_37 = transpose_37 = None
        view_36 = l_self_modules_backbone_modules_layers_modules_18_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_backbone_modules_layers_modules_18_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_78 = l_self_modules_backbone_modules_layers_modules_18_modules_attn_parameters_relative_position_bias_table_[
            view_36
        ]
        l_self_modules_backbone_modules_layers_modules_18_modules_attn_parameters_relative_position_bias_table_ = (
            view_36
        ) = None
        relative_position_bias_36 = getitem_78.view(1601, 1601, -1)
        getitem_78 = None
        permute_40 = relative_position_bias_36.permute(2, 0, 1)
        relative_position_bias_36 = None
        relative_position_bias_37 = permute_40.contiguous()
        permute_40 = None
        unsqueeze_18 = relative_position_bias_37.unsqueeze(0)
        relative_position_bias_37 = None
        attn_73 = attn_72 + unsqueeze_18
        attn_72 = unsqueeze_18 = None
        attn_74 = attn_73.softmax(dim=-1)
        attn_73 = None
        attn_75 = torch.nn.functional.dropout(attn_74, 0.0, False, False)
        attn_74 = None
        matmul_37 = attn_75 @ v_18
        attn_75 = v_18 = None
        transpose_38 = matmul_37.transpose(1, 2)
        matmul_37 = None
        x_93 = transpose_38.reshape(1, 1601, 1024)
        transpose_38 = None
        x_94 = torch._C._nn.linear(
            x_93,
            l_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_proj_parameters_bias_,
        )
        x_93 = l_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_proj_parameters_bias_ = (None)
        x_95 = torch.nn.functional.dropout(x_94, 0.0, False, False)
        x_94 = None
        mul_55 = (
            l_self_modules_backbone_modules_layers_modules_18_parameters_gamma_1_ * x_95
        )
        l_self_modules_backbone_modules_layers_modules_18_parameters_gamma_1_ = (
            x_95
        ) = None
        x_96 = x_92 + mul_55
        x_92 = mul_55 = None
        layer_norm_37 = torch.nn.functional.layer_norm(
            x_96,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_18_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_18_modules_ln2_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_18_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_18_modules_ln2_parameters_bias_ = (None)
        input_91 = torch._C._nn.linear(
            layer_norm_37,
            l_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_37 = l_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_92 = torch._C._nn.gelu(input_91, approximate="none")
        input_91 = None
        input_93 = torch.nn.functional.dropout(input_92, 0.0, False, False)
        input_92 = None
        input_94 = torch._C._nn.linear(
            input_93,
            l_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_93 = l_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_95 = torch.nn.functional.dropout(input_94, 0.0, False, False)
        input_94 = None
        mul_56 = (
            l_self_modules_backbone_modules_layers_modules_18_parameters_gamma_2_
            * input_95
        )
        l_self_modules_backbone_modules_layers_modules_18_parameters_gamma_2_ = (
            input_95
        ) = None
        x_97 = x_96 + mul_56
        x_96 = mul_56 = None
        layer_norm_38 = torch.nn.functional.layer_norm(
            x_97,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_19_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_19_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_19_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_19_modules_ln1_parameters_bias_ = (None)
        k_bias_19 = torch.zeros_like(
            l_self_modules_backbone_modules_layers_modules_19_modules_attn_parameters_v_bias_,
            requires_grad=False,
        )
        qkv_bias_19 = torch.cat(
            (
                l_self_modules_backbone_modules_layers_modules_19_modules_attn_parameters_q_bias_,
                k_bias_19,
                l_self_modules_backbone_modules_layers_modules_19_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_backbone_modules_layers_modules_19_modules_attn_parameters_q_bias_ = (
            k_bias_19
        ) = l_self_modules_backbone_modules_layers_modules_19_modules_attn_parameters_v_bias_ = (None)
        qkv_38 = torch._C._nn.linear(
            input=layer_norm_38,
            weight=l_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_19,
        )
        layer_norm_38 = l_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_qkv_parameters_weight_ = (qkv_bias_19) = (
            None
        )
        reshape_41 = qkv_38.reshape(1, 1601, 3, 16, -1)
        qkv_38 = None
        qkv_39 = reshape_41.permute(2, 0, 3, 1, 4)
        reshape_41 = None
        q_38 = qkv_39[0]
        k_19 = qkv_39[1]
        v_19 = qkv_39[2]
        qkv_39 = None
        q_39 = q_38 * 0.125
        q_38 = None
        transpose_39 = k_19.transpose(-2, -1)
        k_19 = None
        attn_76 = q_39 @ transpose_39
        q_39 = transpose_39 = None
        view_38 = l_self_modules_backbone_modules_layers_modules_19_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_backbone_modules_layers_modules_19_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_82 = l_self_modules_backbone_modules_layers_modules_19_modules_attn_parameters_relative_position_bias_table_[
            view_38
        ]
        l_self_modules_backbone_modules_layers_modules_19_modules_attn_parameters_relative_position_bias_table_ = (
            view_38
        ) = None
        relative_position_bias_38 = getitem_82.view(1601, 1601, -1)
        getitem_82 = None
        permute_42 = relative_position_bias_38.permute(2, 0, 1)
        relative_position_bias_38 = None
        relative_position_bias_39 = permute_42.contiguous()
        permute_42 = None
        unsqueeze_19 = relative_position_bias_39.unsqueeze(0)
        relative_position_bias_39 = None
        attn_77 = attn_76 + unsqueeze_19
        attn_76 = unsqueeze_19 = None
        attn_78 = attn_77.softmax(dim=-1)
        attn_77 = None
        attn_79 = torch.nn.functional.dropout(attn_78, 0.0, False, False)
        attn_78 = None
        matmul_39 = attn_79 @ v_19
        attn_79 = v_19 = None
        transpose_40 = matmul_39.transpose(1, 2)
        matmul_39 = None
        x_98 = transpose_40.reshape(1, 1601, 1024)
        transpose_40 = None
        x_99 = torch._C._nn.linear(
            x_98,
            l_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_proj_parameters_bias_,
        )
        x_98 = l_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_proj_parameters_bias_ = (None)
        x_100 = torch.nn.functional.dropout(x_99, 0.0, False, False)
        x_99 = None
        mul_58 = (
            l_self_modules_backbone_modules_layers_modules_19_parameters_gamma_1_
            * x_100
        )
        l_self_modules_backbone_modules_layers_modules_19_parameters_gamma_1_ = (
            x_100
        ) = None
        x_101 = x_97 + mul_58
        x_97 = mul_58 = None
        layer_norm_39 = torch.nn.functional.layer_norm(
            x_101,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_19_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_19_modules_ln2_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_19_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_19_modules_ln2_parameters_bias_ = (None)
        input_96 = torch._C._nn.linear(
            layer_norm_39,
            l_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_39 = l_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_97 = torch._C._nn.gelu(input_96, approximate="none")
        input_96 = None
        input_98 = torch.nn.functional.dropout(input_97, 0.0, False, False)
        input_97 = None
        input_99 = torch._C._nn.linear(
            input_98,
            l_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_98 = l_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_100 = torch.nn.functional.dropout(input_99, 0.0, False, False)
        input_99 = None
        mul_59 = (
            l_self_modules_backbone_modules_layers_modules_19_parameters_gamma_2_
            * input_100
        )
        l_self_modules_backbone_modules_layers_modules_19_parameters_gamma_2_ = (
            input_100
        ) = None
        x_102 = x_101 + mul_59
        x_101 = mul_59 = None
        layer_norm_40 = torch.nn.functional.layer_norm(
            x_102,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_20_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_20_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_20_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_20_modules_ln1_parameters_bias_ = (None)
        k_bias_20 = torch.zeros_like(
            l_self_modules_backbone_modules_layers_modules_20_modules_attn_parameters_v_bias_,
            requires_grad=False,
        )
        qkv_bias_20 = torch.cat(
            (
                l_self_modules_backbone_modules_layers_modules_20_modules_attn_parameters_q_bias_,
                k_bias_20,
                l_self_modules_backbone_modules_layers_modules_20_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_backbone_modules_layers_modules_20_modules_attn_parameters_q_bias_ = (
            k_bias_20
        ) = l_self_modules_backbone_modules_layers_modules_20_modules_attn_parameters_v_bias_ = (None)
        qkv_40 = torch._C._nn.linear(
            input=layer_norm_40,
            weight=l_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_20,
        )
        layer_norm_40 = l_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_qkv_parameters_weight_ = (qkv_bias_20) = (
            None
        )
        reshape_43 = qkv_40.reshape(1, 1601, 3, 16, -1)
        qkv_40 = None
        qkv_41 = reshape_43.permute(2, 0, 3, 1, 4)
        reshape_43 = None
        q_40 = qkv_41[0]
        k_20 = qkv_41[1]
        v_20 = qkv_41[2]
        qkv_41 = None
        q_41 = q_40 * 0.125
        q_40 = None
        transpose_41 = k_20.transpose(-2, -1)
        k_20 = None
        attn_80 = q_41 @ transpose_41
        q_41 = transpose_41 = None
        view_40 = l_self_modules_backbone_modules_layers_modules_20_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_backbone_modules_layers_modules_20_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_86 = l_self_modules_backbone_modules_layers_modules_20_modules_attn_parameters_relative_position_bias_table_[
            view_40
        ]
        l_self_modules_backbone_modules_layers_modules_20_modules_attn_parameters_relative_position_bias_table_ = (
            view_40
        ) = None
        relative_position_bias_40 = getitem_86.view(1601, 1601, -1)
        getitem_86 = None
        permute_44 = relative_position_bias_40.permute(2, 0, 1)
        relative_position_bias_40 = None
        relative_position_bias_41 = permute_44.contiguous()
        permute_44 = None
        unsqueeze_20 = relative_position_bias_41.unsqueeze(0)
        relative_position_bias_41 = None
        attn_81 = attn_80 + unsqueeze_20
        attn_80 = unsqueeze_20 = None
        attn_82 = attn_81.softmax(dim=-1)
        attn_81 = None
        attn_83 = torch.nn.functional.dropout(attn_82, 0.0, False, False)
        attn_82 = None
        matmul_41 = attn_83 @ v_20
        attn_83 = v_20 = None
        transpose_42 = matmul_41.transpose(1, 2)
        matmul_41 = None
        x_103 = transpose_42.reshape(1, 1601, 1024)
        transpose_42 = None
        x_104 = torch._C._nn.linear(
            x_103,
            l_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_proj_parameters_bias_,
        )
        x_103 = l_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_proj_parameters_bias_ = (None)
        x_105 = torch.nn.functional.dropout(x_104, 0.0, False, False)
        x_104 = None
        mul_61 = (
            l_self_modules_backbone_modules_layers_modules_20_parameters_gamma_1_
            * x_105
        )
        l_self_modules_backbone_modules_layers_modules_20_parameters_gamma_1_ = (
            x_105
        ) = None
        x_106 = x_102 + mul_61
        x_102 = mul_61 = None
        layer_norm_41 = torch.nn.functional.layer_norm(
            x_106,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_20_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_20_modules_ln2_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_20_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_20_modules_ln2_parameters_bias_ = (None)
        input_101 = torch._C._nn.linear(
            layer_norm_41,
            l_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_41 = l_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_102 = torch._C._nn.gelu(input_101, approximate="none")
        input_101 = None
        input_103 = torch.nn.functional.dropout(input_102, 0.0, False, False)
        input_102 = None
        input_104 = torch._C._nn.linear(
            input_103,
            l_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_103 = l_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_105 = torch.nn.functional.dropout(input_104, 0.0, False, False)
        input_104 = None
        mul_62 = (
            l_self_modules_backbone_modules_layers_modules_20_parameters_gamma_2_
            * input_105
        )
        l_self_modules_backbone_modules_layers_modules_20_parameters_gamma_2_ = (
            input_105
        ) = None
        x_107 = x_106 + mul_62
        x_106 = mul_62 = None
        layer_norm_42 = torch.nn.functional.layer_norm(
            x_107,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_21_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_21_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_21_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_21_modules_ln1_parameters_bias_ = (None)
        k_bias_21 = torch.zeros_like(
            l_self_modules_backbone_modules_layers_modules_21_modules_attn_parameters_v_bias_,
            requires_grad=False,
        )
        qkv_bias_21 = torch.cat(
            (
                l_self_modules_backbone_modules_layers_modules_21_modules_attn_parameters_q_bias_,
                k_bias_21,
                l_self_modules_backbone_modules_layers_modules_21_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_backbone_modules_layers_modules_21_modules_attn_parameters_q_bias_ = (
            k_bias_21
        ) = l_self_modules_backbone_modules_layers_modules_21_modules_attn_parameters_v_bias_ = (None)
        qkv_42 = torch._C._nn.linear(
            input=layer_norm_42,
            weight=l_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_21,
        )
        layer_norm_42 = l_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_qkv_parameters_weight_ = (qkv_bias_21) = (
            None
        )
        reshape_45 = qkv_42.reshape(1, 1601, 3, 16, -1)
        qkv_42 = None
        qkv_43 = reshape_45.permute(2, 0, 3, 1, 4)
        reshape_45 = None
        q_42 = qkv_43[0]
        k_21 = qkv_43[1]
        v_21 = qkv_43[2]
        qkv_43 = None
        q_43 = q_42 * 0.125
        q_42 = None
        transpose_43 = k_21.transpose(-2, -1)
        k_21 = None
        attn_84 = q_43 @ transpose_43
        q_43 = transpose_43 = None
        view_42 = l_self_modules_backbone_modules_layers_modules_21_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_backbone_modules_layers_modules_21_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_90 = l_self_modules_backbone_modules_layers_modules_21_modules_attn_parameters_relative_position_bias_table_[
            view_42
        ]
        l_self_modules_backbone_modules_layers_modules_21_modules_attn_parameters_relative_position_bias_table_ = (
            view_42
        ) = None
        relative_position_bias_42 = getitem_90.view(1601, 1601, -1)
        getitem_90 = None
        permute_46 = relative_position_bias_42.permute(2, 0, 1)
        relative_position_bias_42 = None
        relative_position_bias_43 = permute_46.contiguous()
        permute_46 = None
        unsqueeze_21 = relative_position_bias_43.unsqueeze(0)
        relative_position_bias_43 = None
        attn_85 = attn_84 + unsqueeze_21
        attn_84 = unsqueeze_21 = None
        attn_86 = attn_85.softmax(dim=-1)
        attn_85 = None
        attn_87 = torch.nn.functional.dropout(attn_86, 0.0, False, False)
        attn_86 = None
        matmul_43 = attn_87 @ v_21
        attn_87 = v_21 = None
        transpose_44 = matmul_43.transpose(1, 2)
        matmul_43 = None
        x_108 = transpose_44.reshape(1, 1601, 1024)
        transpose_44 = None
        x_109 = torch._C._nn.linear(
            x_108,
            l_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_proj_parameters_bias_,
        )
        x_108 = l_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_proj_parameters_bias_ = (None)
        x_110 = torch.nn.functional.dropout(x_109, 0.0, False, False)
        x_109 = None
        mul_64 = (
            l_self_modules_backbone_modules_layers_modules_21_parameters_gamma_1_
            * x_110
        )
        l_self_modules_backbone_modules_layers_modules_21_parameters_gamma_1_ = (
            x_110
        ) = None
        x_111 = x_107 + mul_64
        x_107 = mul_64 = None
        layer_norm_43 = torch.nn.functional.layer_norm(
            x_111,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_21_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_21_modules_ln2_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_21_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_21_modules_ln2_parameters_bias_ = (None)
        input_106 = torch._C._nn.linear(
            layer_norm_43,
            l_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_43 = l_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_107 = torch._C._nn.gelu(input_106, approximate="none")
        input_106 = None
        input_108 = torch.nn.functional.dropout(input_107, 0.0, False, False)
        input_107 = None
        input_109 = torch._C._nn.linear(
            input_108,
            l_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_108 = l_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_110 = torch.nn.functional.dropout(input_109, 0.0, False, False)
        input_109 = None
        mul_65 = (
            l_self_modules_backbone_modules_layers_modules_21_parameters_gamma_2_
            * input_110
        )
        l_self_modules_backbone_modules_layers_modules_21_parameters_gamma_2_ = (
            input_110
        ) = None
        x_112 = x_111 + mul_65
        x_111 = mul_65 = None
        layer_norm_44 = torch.nn.functional.layer_norm(
            x_112,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_22_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_22_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_22_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_22_modules_ln1_parameters_bias_ = (None)
        k_bias_22 = torch.zeros_like(
            l_self_modules_backbone_modules_layers_modules_22_modules_attn_parameters_v_bias_,
            requires_grad=False,
        )
        qkv_bias_22 = torch.cat(
            (
                l_self_modules_backbone_modules_layers_modules_22_modules_attn_parameters_q_bias_,
                k_bias_22,
                l_self_modules_backbone_modules_layers_modules_22_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_backbone_modules_layers_modules_22_modules_attn_parameters_q_bias_ = (
            k_bias_22
        ) = l_self_modules_backbone_modules_layers_modules_22_modules_attn_parameters_v_bias_ = (None)
        qkv_44 = torch._C._nn.linear(
            input=layer_norm_44,
            weight=l_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_22,
        )
        layer_norm_44 = l_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_qkv_parameters_weight_ = (qkv_bias_22) = (
            None
        )
        reshape_47 = qkv_44.reshape(1, 1601, 3, 16, -1)
        qkv_44 = None
        qkv_45 = reshape_47.permute(2, 0, 3, 1, 4)
        reshape_47 = None
        q_44 = qkv_45[0]
        k_22 = qkv_45[1]
        v_22 = qkv_45[2]
        qkv_45 = None
        q_45 = q_44 * 0.125
        q_44 = None
        transpose_45 = k_22.transpose(-2, -1)
        k_22 = None
        attn_88 = q_45 @ transpose_45
        q_45 = transpose_45 = None
        view_44 = l_self_modules_backbone_modules_layers_modules_22_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_backbone_modules_layers_modules_22_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_94 = l_self_modules_backbone_modules_layers_modules_22_modules_attn_parameters_relative_position_bias_table_[
            view_44
        ]
        l_self_modules_backbone_modules_layers_modules_22_modules_attn_parameters_relative_position_bias_table_ = (
            view_44
        ) = None
        relative_position_bias_44 = getitem_94.view(1601, 1601, -1)
        getitem_94 = None
        permute_48 = relative_position_bias_44.permute(2, 0, 1)
        relative_position_bias_44 = None
        relative_position_bias_45 = permute_48.contiguous()
        permute_48 = None
        unsqueeze_22 = relative_position_bias_45.unsqueeze(0)
        relative_position_bias_45 = None
        attn_89 = attn_88 + unsqueeze_22
        attn_88 = unsqueeze_22 = None
        attn_90 = attn_89.softmax(dim=-1)
        attn_89 = None
        attn_91 = torch.nn.functional.dropout(attn_90, 0.0, False, False)
        attn_90 = None
        matmul_45 = attn_91 @ v_22
        attn_91 = v_22 = None
        transpose_46 = matmul_45.transpose(1, 2)
        matmul_45 = None
        x_113 = transpose_46.reshape(1, 1601, 1024)
        transpose_46 = None
        x_114 = torch._C._nn.linear(
            x_113,
            l_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_proj_parameters_bias_,
        )
        x_113 = l_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_proj_parameters_bias_ = (None)
        x_115 = torch.nn.functional.dropout(x_114, 0.0, False, False)
        x_114 = None
        mul_67 = (
            l_self_modules_backbone_modules_layers_modules_22_parameters_gamma_1_
            * x_115
        )
        l_self_modules_backbone_modules_layers_modules_22_parameters_gamma_1_ = (
            x_115
        ) = None
        x_116 = x_112 + mul_67
        x_112 = mul_67 = None
        layer_norm_45 = torch.nn.functional.layer_norm(
            x_116,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_22_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_22_modules_ln2_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_22_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_22_modules_ln2_parameters_bias_ = (None)
        input_111 = torch._C._nn.linear(
            layer_norm_45,
            l_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_45 = l_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_112 = torch._C._nn.gelu(input_111, approximate="none")
        input_111 = None
        input_113 = torch.nn.functional.dropout(input_112, 0.0, False, False)
        input_112 = None
        input_114 = torch._C._nn.linear(
            input_113,
            l_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_113 = l_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_115 = torch.nn.functional.dropout(input_114, 0.0, False, False)
        input_114 = None
        mul_68 = (
            l_self_modules_backbone_modules_layers_modules_22_parameters_gamma_2_
            * input_115
        )
        l_self_modules_backbone_modules_layers_modules_22_parameters_gamma_2_ = (
            input_115
        ) = None
        x_117 = x_116 + mul_68
        x_116 = mul_68 = None
        layer_norm_46 = torch.nn.functional.layer_norm(
            x_117,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_23_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_23_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_23_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_23_modules_ln1_parameters_bias_ = (None)
        k_bias_23 = torch.zeros_like(
            l_self_modules_backbone_modules_layers_modules_23_modules_attn_parameters_v_bias_,
            requires_grad=False,
        )
        qkv_bias_23 = torch.cat(
            (
                l_self_modules_backbone_modules_layers_modules_23_modules_attn_parameters_q_bias_,
                k_bias_23,
                l_self_modules_backbone_modules_layers_modules_23_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_backbone_modules_layers_modules_23_modules_attn_parameters_q_bias_ = (
            k_bias_23
        ) = l_self_modules_backbone_modules_layers_modules_23_modules_attn_parameters_v_bias_ = (None)
        qkv_46 = torch._C._nn.linear(
            input=layer_norm_46,
            weight=l_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_23,
        )
        layer_norm_46 = l_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_qkv_parameters_weight_ = (qkv_bias_23) = (
            None
        )
        reshape_49 = qkv_46.reshape(1, 1601, 3, 16, -1)
        qkv_46 = None
        qkv_47 = reshape_49.permute(2, 0, 3, 1, 4)
        reshape_49 = None
        q_46 = qkv_47[0]
        k_23 = qkv_47[1]
        v_23 = qkv_47[2]
        qkv_47 = None
        q_47 = q_46 * 0.125
        q_46 = None
        transpose_47 = k_23.transpose(-2, -1)
        k_23 = None
        attn_92 = q_47 @ transpose_47
        q_47 = transpose_47 = None
        view_46 = l_self_modules_backbone_modules_layers_modules_23_modules_attn_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_backbone_modules_layers_modules_23_modules_attn_buffers_relative_position_index_ = (
            None
        )
        getitem_98 = l_self_modules_backbone_modules_layers_modules_23_modules_attn_parameters_relative_position_bias_table_[
            view_46
        ]
        l_self_modules_backbone_modules_layers_modules_23_modules_attn_parameters_relative_position_bias_table_ = (
            view_46
        ) = None
        relative_position_bias_46 = getitem_98.view(1601, 1601, -1)
        getitem_98 = None
        permute_50 = relative_position_bias_46.permute(2, 0, 1)
        relative_position_bias_46 = None
        relative_position_bias_47 = permute_50.contiguous()
        permute_50 = None
        unsqueeze_23 = relative_position_bias_47.unsqueeze(0)
        relative_position_bias_47 = None
        attn_93 = attn_92 + unsqueeze_23
        attn_92 = unsqueeze_23 = None
        attn_94 = attn_93.softmax(dim=-1)
        attn_93 = None
        attn_95 = torch.nn.functional.dropout(attn_94, 0.0, False, False)
        attn_94 = None
        matmul_47 = attn_95 @ v_23
        attn_95 = v_23 = None
        transpose_48 = matmul_47.transpose(1, 2)
        matmul_47 = None
        x_118 = transpose_48.reshape(1, 1601, 1024)
        transpose_48 = None
        x_119 = torch._C._nn.linear(
            x_118,
            l_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_proj_parameters_bias_,
        )
        x_118 = l_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_proj_parameters_bias_ = (None)
        x_120 = torch.nn.functional.dropout(x_119, 0.0, False, False)
        x_119 = None
        mul_70 = (
            l_self_modules_backbone_modules_layers_modules_23_parameters_gamma_1_
            * x_120
        )
        l_self_modules_backbone_modules_layers_modules_23_parameters_gamma_1_ = (
            x_120
        ) = None
        x_121 = x_117 + mul_70
        x_117 = mul_70 = None
        layer_norm_47 = torch.nn.functional.layer_norm(
            x_121,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_23_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_23_modules_ln2_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_23_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_23_modules_ln2_parameters_bias_ = (None)
        input_116 = torch._C._nn.linear(
            layer_norm_47,
            l_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_47 = l_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_117 = torch._C._nn.gelu(input_116, approximate="none")
        input_116 = None
        input_118 = torch.nn.functional.dropout(input_117, 0.0, False, False)
        input_117 = None
        input_119 = torch._C._nn.linear(
            input_118,
            l_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_118 = l_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_120 = torch.nn.functional.dropout(input_119, 0.0, False, False)
        input_119 = None
        mul_71 = (
            l_self_modules_backbone_modules_layers_modules_23_parameters_gamma_2_
            * input_120
        )
        l_self_modules_backbone_modules_layers_modules_23_parameters_gamma_2_ = (
            input_120
        ) = None
        x_122 = x_121 + mul_71
        x_121 = mul_71 = None
        out_6 = x_122[(slice(None, None, None), slice(1, None, None))]
        x_122 = None
        reshape_51 = out_6.reshape(1, 40, 40, 1024)
        out_6 = None
        permute_51 = reshape_51.permute(0, 3, 1, 2)
        reshape_51 = None
        out_7 = permute_51.contiguous()
        permute_51 = None
        input_121 = torch.conv_transpose2d(
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
        input_122 = torch.nn.functional.batch_norm(
            input_121,
            l_self_modules_neck_modules_upsample_4x_modules_1_buffers_running_mean_,
            l_self_modules_neck_modules_upsample_4x_modules_1_buffers_running_var_,
            l_self_modules_neck_modules_upsample_4x_modules_1_parameters_weight_,
            l_self_modules_neck_modules_upsample_4x_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_121 = (
            l_self_modules_neck_modules_upsample_4x_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_neck_modules_upsample_4x_modules_1_buffers_running_var_
        ) = (
            l_self_modules_neck_modules_upsample_4x_modules_1_parameters_weight_
        ) = l_self_modules_neck_modules_upsample_4x_modules_1_parameters_bias_ = None
        input_123 = torch._C._nn.gelu(input_122, approximate="none")
        input_122 = None
        input_124 = torch.conv_transpose2d(
            input_123,
            l_self_modules_neck_modules_upsample_4x_modules_3_parameters_weight_,
            l_self_modules_neck_modules_upsample_4x_modules_3_parameters_bias_,
            (2, 2),
            (0, 0),
            (0, 0),
            1,
            (1, 1),
        )
        input_123 = (
            l_self_modules_neck_modules_upsample_4x_modules_3_parameters_weight_
        ) = l_self_modules_neck_modules_upsample_4x_modules_3_parameters_bias_ = None
        input_125 = torch.conv_transpose2d(
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
        x_132 = torch.nn.functional.max_pool2d(
            out_7, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        out_7 = None
        x_123 = torch.conv2d(
            input_124,
            l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_124 = l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_conv_parameters_weight_ = (None)
        x_124 = torch.nn.functional.batch_norm(
            x_123,
            l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_123 = l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_bias_ = (None)
        x_125 = torch.nn.functional.relu(x_124, inplace=False)
        x_124 = None
        x_126 = torch.conv2d(
            input_125,
            l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_125 = l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_conv_parameters_weight_ = (None)
        x_127 = torch.nn.functional.batch_norm(
            x_126,
            l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_126 = l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_bias_ = (None)
        x_128 = torch.nn.functional.relu(x_127, inplace=False)
        x_127 = None
        x_129 = torch.conv2d(
            out_5,
            l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_5 = l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_conv_parameters_weight_ = (None)
        x_130 = torch.nn.functional.batch_norm(
            x_129,
            l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_129 = l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_bias_ = (None)
        x_131 = torch.nn.functional.relu(x_130, inplace=False)
        x_130 = None
        input_126 = torch.nn.functional.adaptive_avg_pool2d(x_132, 1)
        x_133 = torch.conv2d(
            input_126,
            l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_126 = l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_conv_parameters_weight_ = (None)
        x_134 = torch.nn.functional.batch_norm(
            x_133,
            l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_133 = l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_bias_ = (None)
        x_135 = torch.nn.functional.relu(x_134, inplace=True)
        x_134 = None
        upsampled_ppm_out = torch.nn.functional.interpolate(
            x_135, (20, 20), None, "bilinear", False
        )
        x_135 = None
        input_127 = torch.nn.functional.adaptive_avg_pool2d(x_132, 2)
        x_136 = torch.conv2d(
            input_127,
            l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_127 = l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_conv_parameters_weight_ = (None)
        x_137 = torch.nn.functional.batch_norm(
            x_136,
            l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_136 = l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_bias_ = (None)
        x_138 = torch.nn.functional.relu(x_137, inplace=True)
        x_137 = None
        upsampled_ppm_out_1 = torch.nn.functional.interpolate(
            x_138, (20, 20), None, "bilinear", False
        )
        x_138 = None
        input_128 = torch.nn.functional.adaptive_avg_pool2d(x_132, 3)
        x_139 = torch.conv2d(
            input_128,
            l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_128 = l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_conv_parameters_weight_ = (None)
        x_140 = torch.nn.functional.batch_norm(
            x_139,
            l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_139 = l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_bias_ = (None)
        x_141 = torch.nn.functional.relu(x_140, inplace=True)
        x_140 = None
        upsampled_ppm_out_2 = torch.nn.functional.interpolate(
            x_141, (20, 20), None, "bilinear", False
        )
        x_141 = None
        input_129 = torch.nn.functional.adaptive_avg_pool2d(x_132, 6)
        x_142 = torch.conv2d(
            input_129,
            l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_129 = l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_conv_parameters_weight_ = (None)
        x_143 = torch.nn.functional.batch_norm(
            x_142,
            l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_142 = l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_bias_ = (None)
        x_144 = torch.nn.functional.relu(x_143, inplace=True)
        x_143 = None
        upsampled_ppm_out_3 = torch.nn.functional.interpolate(
            x_144, (20, 20), None, "bilinear", False
        )
        x_144 = None
        psp_outs = torch.cat(
            [
                x_132,
                upsampled_ppm_out,
                upsampled_ppm_out_1,
                upsampled_ppm_out_2,
                upsampled_ppm_out_3,
            ],
            dim=1,
        )
        x_132 = (
            upsampled_ppm_out
        ) = upsampled_ppm_out_1 = upsampled_ppm_out_2 = upsampled_ppm_out_3 = None
        x_145 = torch.conv2d(
            psp_outs,
            l_self_modules_decode_head_modules_bottleneck_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        psp_outs = l_self_modules_decode_head_modules_bottleneck_modules_conv_parameters_weight_ = (None)
        x_146 = torch.nn.functional.batch_norm(
            x_145,
            l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_145 = l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_var_ = (
            l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_weight_
        ) = (
            l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_bias_
        ) = None
        x_147 = torch.nn.functional.relu(x_146, inplace=True)
        x_146 = None
        interpolate_4 = torch.nn.functional.interpolate(
            x_147, (40, 40), None, "bilinear", False
        )
        add_72 = x_131 + interpolate_4
        x_131 = interpolate_4 = None
        interpolate_5 = torch.nn.functional.interpolate(
            add_72, (80, 80), None, "bilinear", False
        )
        add_73 = x_128 + interpolate_5
        x_128 = interpolate_5 = None
        interpolate_6 = torch.nn.functional.interpolate(
            add_73, (160, 160), None, "bilinear", False
        )
        add_74 = x_125 + interpolate_6
        x_125 = interpolate_6 = None
        x_148 = torch.conv2d(
            add_74,
            l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        add_74 = l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_conv_parameters_weight_ = (None)
        x_149 = torch.nn.functional.batch_norm(
            x_148,
            l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_148 = l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_bias_ = (None)
        x_150 = torch.nn.functional.relu(x_149, inplace=False)
        x_149 = None
        x_151 = torch.conv2d(
            add_73,
            l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        add_73 = l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_conv_parameters_weight_ = (None)
        x_152 = torch.nn.functional.batch_norm(
            x_151,
            l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_151 = l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_bias_ = (None)
        x_153 = torch.nn.functional.relu(x_152, inplace=False)
        x_152 = None
        x_154 = torch.conv2d(
            add_72,
            l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        add_72 = l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_conv_parameters_weight_ = (None)
        x_155 = torch.nn.functional.batch_norm(
            x_154,
            l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_154 = l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_bias_ = (None)
        x_156 = torch.nn.functional.relu(x_155, inplace=False)
        x_155 = None
        interpolate_7 = torch.nn.functional.interpolate(
            x_147, (160, 160), None, "bilinear", False
        )
        x_147 = None
        interpolate_8 = torch.nn.functional.interpolate(
            x_156, (160, 160), None, "bilinear", False
        )
        x_156 = None
        interpolate_9 = torch.nn.functional.interpolate(
            x_153, (160, 160), None, "bilinear", False
        )
        x_153 = None
        fpn_outs = torch.cat(
            [x_150, interpolate_9, interpolate_8, interpolate_7], dim=1
        )
        x_150 = interpolate_9 = interpolate_8 = interpolate_7 = None
        x_157 = torch.conv2d(
            fpn_outs,
            l_self_modules_decode_head_modules_fpn_bottleneck_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        fpn_outs = l_self_modules_decode_head_modules_fpn_bottleneck_modules_conv_parameters_weight_ = (None)
        x_158 = torch.nn.functional.batch_norm(
            x_157,
            l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_157 = l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_bias_ = (None)
        x_159 = torch.nn.functional.relu(x_158, inplace=True)
        x_158 = None
        feat = torch.nn.functional.dropout2d(x_159, 0.1, False, False)
        x_159 = None
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
