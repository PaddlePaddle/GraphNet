import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_inputs_: torch.Tensor,
        L_self_modules_backbone_modules_patch_embed_modules_projection_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embed_modules_projection_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_parameters_pos_embed_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_12_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_12_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_12_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_12_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_24_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_24_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_24_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_24_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_24_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_24_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_24_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_24_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_24_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_24_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_24_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_24_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_25_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_25_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_25_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_25_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_25_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_25_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_25_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_25_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_25_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_25_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_25_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_25_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_26_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_26_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_26_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_26_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_26_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_26_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_26_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_26_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_26_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_26_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_26_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_26_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_27_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_27_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_27_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_27_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_27_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_27_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_27_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_27_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_27_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_27_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_27_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_27_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_28_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_28_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_28_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_28_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_28_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_28_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_28_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_28_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_28_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_28_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_28_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_28_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_29_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_29_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_29_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_29_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_29_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_29_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_29_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_29_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_29_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_29_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_29_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_29_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_30_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_30_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_30_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_30_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_30_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_30_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_30_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_30_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_30_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_30_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_30_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_30_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_31_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_31_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_31_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_31_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_31_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_31_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_31_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_31_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_31_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_31_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_31_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_31_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_deconv_layers_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_deconv_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_deconv_layers_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_deconv_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_final_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_final_layer_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_inputs_ = L_inputs_
        l_self_modules_backbone_modules_patch_embed_modules_projection_parameters_weight_ = L_self_modules_backbone_modules_patch_embed_modules_projection_parameters_weight_
        l_self_modules_backbone_modules_patch_embed_modules_projection_parameters_bias_ = L_self_modules_backbone_modules_patch_embed_modules_projection_parameters_bias_
        l_self_modules_backbone_parameters_pos_embed_ = (
            L_self_modules_backbone_parameters_pos_embed_
        )
        l_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_4_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_4_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_4_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_4_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_5_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_5_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_5_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_5_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_6_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_6_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_6_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_6_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_7_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_7_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_7_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_7_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_8_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_8_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_8_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_8_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_9_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_9_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_9_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_9_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_10_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_10_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_10_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_10_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_11_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_11_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_11_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_11_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_12_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_12_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_12_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_12_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_12_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_12_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_12_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_12_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_13_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_13_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_13_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_13_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_13_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_13_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_13_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_13_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_14_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_14_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_14_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_14_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_14_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_14_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_14_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_14_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_15_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_15_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_15_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_15_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_15_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_15_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_15_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_15_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_16_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_16_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_16_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_16_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_16_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_16_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_16_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_16_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_17_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_17_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_17_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_17_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_17_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_17_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_17_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_17_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_18_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_18_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_18_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_18_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_18_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_18_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_18_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_18_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_19_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_19_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_19_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_19_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_19_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_19_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_19_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_19_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_20_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_20_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_20_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_20_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_20_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_20_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_20_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_20_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_21_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_21_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_21_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_21_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_21_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_21_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_21_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_21_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_22_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_22_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_22_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_22_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_22_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_22_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_22_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_22_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_23_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_23_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_23_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_23_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_23_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_23_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_23_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_23_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_24_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_24_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_24_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_24_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_24_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_24_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_24_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_24_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_24_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_24_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_24_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_24_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_24_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_24_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_24_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_24_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_24_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_24_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_24_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_24_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_24_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_24_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_24_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_24_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_25_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_25_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_25_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_25_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_25_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_25_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_25_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_25_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_25_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_25_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_25_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_25_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_25_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_25_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_25_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_25_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_25_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_25_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_25_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_25_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_25_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_25_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_25_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_25_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_26_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_26_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_26_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_26_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_26_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_26_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_26_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_26_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_26_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_26_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_26_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_26_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_26_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_26_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_26_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_26_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_26_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_26_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_26_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_26_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_26_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_26_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_26_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_26_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_27_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_27_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_27_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_27_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_27_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_27_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_27_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_27_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_27_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_27_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_27_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_27_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_27_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_27_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_27_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_27_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_27_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_27_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_27_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_27_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_27_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_27_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_27_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_27_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_28_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_28_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_28_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_28_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_28_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_28_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_28_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_28_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_28_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_28_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_28_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_28_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_28_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_28_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_28_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_28_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_28_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_28_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_28_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_28_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_28_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_28_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_28_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_28_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_29_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_29_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_29_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_29_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_29_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_29_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_29_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_29_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_29_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_29_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_29_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_29_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_29_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_29_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_29_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_29_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_29_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_29_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_29_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_29_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_29_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_29_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_29_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_29_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_30_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_30_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_30_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_30_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_30_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_30_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_30_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_30_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_30_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_30_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_30_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_30_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_30_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_30_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_30_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_30_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_30_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_30_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_30_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_30_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_30_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_30_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_30_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_30_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_31_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_31_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_31_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_31_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_31_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_31_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_31_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_31_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_31_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_31_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_31_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_31_modules_attn_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_31_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_31_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_31_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_31_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_31_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_31_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_31_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_31_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_31_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_31_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_31_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_31_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_ln1_parameters_weight_ = (
            L_self_modules_backbone_modules_ln1_parameters_weight_
        )
        l_self_modules_backbone_modules_ln1_parameters_bias_ = (
            L_self_modules_backbone_modules_ln1_parameters_bias_
        )
        l_self_modules_head_modules_deconv_layers_modules_0_parameters_weight_ = (
            L_self_modules_head_modules_deconv_layers_modules_0_parameters_weight_
        )
        l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_mean_ = (
            L_self_modules_head_modules_deconv_layers_modules_1_buffers_running_mean_
        )
        l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_var_ = (
            L_self_modules_head_modules_deconv_layers_modules_1_buffers_running_var_
        )
        l_self_modules_head_modules_deconv_layers_modules_1_parameters_weight_ = (
            L_self_modules_head_modules_deconv_layers_modules_1_parameters_weight_
        )
        l_self_modules_head_modules_deconv_layers_modules_1_parameters_bias_ = (
            L_self_modules_head_modules_deconv_layers_modules_1_parameters_bias_
        )
        l_self_modules_head_modules_deconv_layers_modules_3_parameters_weight_ = (
            L_self_modules_head_modules_deconv_layers_modules_3_parameters_weight_
        )
        l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_mean_ = (
            L_self_modules_head_modules_deconv_layers_modules_4_buffers_running_mean_
        )
        l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_var_ = (
            L_self_modules_head_modules_deconv_layers_modules_4_buffers_running_var_
        )
        l_self_modules_head_modules_deconv_layers_modules_4_parameters_weight_ = (
            L_self_modules_head_modules_deconv_layers_modules_4_parameters_weight_
        )
        l_self_modules_head_modules_deconv_layers_modules_4_parameters_bias_ = (
            L_self_modules_head_modules_deconv_layers_modules_4_parameters_bias_
        )
        l_self_modules_head_modules_final_layer_parameters_weight_ = (
            L_self_modules_head_modules_final_layer_parameters_weight_
        )
        l_self_modules_head_modules_final_layer_parameters_bias_ = (
            L_self_modules_head_modules_final_layer_parameters_bias_
        )
        x = torch.conv2d(
            l_inputs_,
            l_self_modules_backbone_modules_patch_embed_modules_projection_parameters_weight_,
            l_self_modules_backbone_modules_patch_embed_modules_projection_parameters_bias_,
            (16, 16),
            (2, 2),
            (1, 1),
            1,
        )
        l_inputs_ = l_self_modules_backbone_modules_patch_embed_modules_projection_parameters_weight_ = l_self_modules_backbone_modules_patch_embed_modules_projection_parameters_bias_ = (None)
        flatten = x.flatten(2)
        x = None
        x_1 = flatten.transpose(1, 2)
        flatten = None
        x_2 = x_1 + l_self_modules_backbone_parameters_pos_embed_
        x_1 = l_self_modules_backbone_parameters_pos_embed_ = None
        x_3 = torch.nn.functional.dropout(x_2, 0.0, False, False)
        x_2 = None
        layer_norm = torch.nn.functional.layer_norm(
            x_3,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_bias_ = (None)
        linear = torch._C._nn.linear(
            layer_norm,
            l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm = l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape = linear.reshape(1, 192, 3, 16, 80)
        linear = None
        qkv = reshape.permute(2, 0, 3, 1, 4)
        reshape = None
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]
        qkv = None
        x_4 = torch._C._nn.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        q = k = v = None
        transpose_1 = x_4.transpose(1, 2)
        x_4 = None
        x_5 = transpose_1.reshape(1, 192, 1280)
        transpose_1 = None
        x_6 = torch._C._nn.linear(
            x_5,
            l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_5 = l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_7 = torch.nn.functional.dropout(x_6, 0.0, False, False)
        x_6 = None
        x_8 = x_3 + x_7
        x_3 = x_7 = None
        layer_norm_1 = torch.nn.functional.layer_norm(
            x_8,
            (1280,),
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
        output = x_8 + input_5
        x_8 = input_5 = None
        layer_norm_2 = torch.nn.functional.layer_norm(
            output,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_bias_ = (None)
        linear_4 = torch._C._nn.linear(
            layer_norm_2,
            l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_2 = l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_2 = linear_4.reshape(1, 192, 3, 16, 80)
        linear_4 = None
        qkv_1 = reshape_2.permute(2, 0, 3, 1, 4)
        reshape_2 = None
        q_1 = qkv_1[0]
        k_1 = qkv_1[1]
        v_1 = qkv_1[2]
        qkv_1 = None
        x_9 = torch._C._nn.scaled_dot_product_attention(q_1, k_1, v_1, dropout_p=0.0)
        q_1 = k_1 = v_1 = None
        transpose_2 = x_9.transpose(1, 2)
        x_9 = None
        x_10 = transpose_2.reshape(1, 192, 1280)
        transpose_2 = None
        x_11 = torch._C._nn.linear(
            x_10,
            l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_10 = l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_12 = torch.nn.functional.dropout(x_11, 0.0, False, False)
        x_11 = None
        x_13 = output + x_12
        output = x_12 = None
        layer_norm_3 = torch.nn.functional.layer_norm(
            x_13,
            (1280,),
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
        output_1 = x_13 + input_10
        x_13 = input_10 = None
        layer_norm_4 = torch.nn.functional.layer_norm(
            output_1,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_bias_ = (None)
        linear_8 = torch._C._nn.linear(
            layer_norm_4,
            l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_4 = l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_4 = linear_8.reshape(1, 192, 3, 16, 80)
        linear_8 = None
        qkv_2 = reshape_4.permute(2, 0, 3, 1, 4)
        reshape_4 = None
        q_2 = qkv_2[0]
        k_2 = qkv_2[1]
        v_2 = qkv_2[2]
        qkv_2 = None
        x_14 = torch._C._nn.scaled_dot_product_attention(q_2, k_2, v_2, dropout_p=0.0)
        q_2 = k_2 = v_2 = None
        transpose_3 = x_14.transpose(1, 2)
        x_14 = None
        x_15 = transpose_3.reshape(1, 192, 1280)
        transpose_3 = None
        x_16 = torch._C._nn.linear(
            x_15,
            l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_proj_parameters_bias_,
        )
        x_15 = l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_proj_parameters_bias_ = (None)
        x_17 = torch.nn.functional.dropout(x_16, 0.0, False, False)
        x_16 = None
        x_18 = output_1 + x_17
        output_1 = x_17 = None
        layer_norm_5 = torch.nn.functional.layer_norm(
            x_18,
            (1280,),
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
        output_2 = x_18 + input_15
        x_18 = input_15 = None
        layer_norm_6 = torch.nn.functional.layer_norm(
            output_2,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_bias_ = (None)
        linear_12 = torch._C._nn.linear(
            layer_norm_6,
            l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_6 = l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_6 = linear_12.reshape(1, 192, 3, 16, 80)
        linear_12 = None
        qkv_3 = reshape_6.permute(2, 0, 3, 1, 4)
        reshape_6 = None
        q_3 = qkv_3[0]
        k_3 = qkv_3[1]
        v_3 = qkv_3[2]
        qkv_3 = None
        x_19 = torch._C._nn.scaled_dot_product_attention(q_3, k_3, v_3, dropout_p=0.0)
        q_3 = k_3 = v_3 = None
        transpose_4 = x_19.transpose(1, 2)
        x_19 = None
        x_20 = transpose_4.reshape(1, 192, 1280)
        transpose_4 = None
        x_21 = torch._C._nn.linear(
            x_20,
            l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_proj_parameters_bias_,
        )
        x_20 = l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_proj_parameters_bias_ = (None)
        x_22 = torch.nn.functional.dropout(x_21, 0.0, False, False)
        x_21 = None
        x_23 = output_2 + x_22
        output_2 = x_22 = None
        layer_norm_7 = torch.nn.functional.layer_norm(
            x_23,
            (1280,),
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
        output_3 = x_23 + input_20
        x_23 = input_20 = None
        layer_norm_8 = torch.nn.functional.layer_norm(
            output_3,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_bias_ = (None)
        linear_16 = torch._C._nn.linear(
            layer_norm_8,
            l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_8 = l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_8 = linear_16.reshape(1, 192, 3, 16, 80)
        linear_16 = None
        qkv_4 = reshape_8.permute(2, 0, 3, 1, 4)
        reshape_8 = None
        q_4 = qkv_4[0]
        k_4 = qkv_4[1]
        v_4 = qkv_4[2]
        qkv_4 = None
        x_24 = torch._C._nn.scaled_dot_product_attention(q_4, k_4, v_4, dropout_p=0.0)
        q_4 = k_4 = v_4 = None
        transpose_5 = x_24.transpose(1, 2)
        x_24 = None
        x_25 = transpose_5.reshape(1, 192, 1280)
        transpose_5 = None
        x_26 = torch._C._nn.linear(
            x_25,
            l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_proj_parameters_bias_,
        )
        x_25 = l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_proj_parameters_bias_ = (None)
        x_27 = torch.nn.functional.dropout(x_26, 0.0, False, False)
        x_26 = None
        x_28 = output_3 + x_27
        output_3 = x_27 = None
        layer_norm_9 = torch.nn.functional.layer_norm(
            x_28,
            (1280,),
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
        output_4 = x_28 + input_25
        x_28 = input_25 = None
        layer_norm_10 = torch.nn.functional.layer_norm(
            output_4,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_bias_ = (None)
        linear_20 = torch._C._nn.linear(
            layer_norm_10,
            l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_10 = l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_10 = linear_20.reshape(1, 192, 3, 16, 80)
        linear_20 = None
        qkv_5 = reshape_10.permute(2, 0, 3, 1, 4)
        reshape_10 = None
        q_5 = qkv_5[0]
        k_5 = qkv_5[1]
        v_5 = qkv_5[2]
        qkv_5 = None
        x_29 = torch._C._nn.scaled_dot_product_attention(q_5, k_5, v_5, dropout_p=0.0)
        q_5 = k_5 = v_5 = None
        transpose_6 = x_29.transpose(1, 2)
        x_29 = None
        x_30 = transpose_6.reshape(1, 192, 1280)
        transpose_6 = None
        x_31 = torch._C._nn.linear(
            x_30,
            l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_proj_parameters_bias_,
        )
        x_30 = l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_proj_parameters_bias_ = (None)
        x_32 = torch.nn.functional.dropout(x_31, 0.0, False, False)
        x_31 = None
        x_33 = output_4 + x_32
        output_4 = x_32 = None
        layer_norm_11 = torch.nn.functional.layer_norm(
            x_33,
            (1280,),
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
        output_5 = x_33 + input_30
        x_33 = input_30 = None
        layer_norm_12 = torch.nn.functional.layer_norm(
            output_5,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_bias_ = (None)
        linear_24 = torch._C._nn.linear(
            layer_norm_12,
            l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_12 = l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_12 = linear_24.reshape(1, 192, 3, 16, 80)
        linear_24 = None
        qkv_6 = reshape_12.permute(2, 0, 3, 1, 4)
        reshape_12 = None
        q_6 = qkv_6[0]
        k_6 = qkv_6[1]
        v_6 = qkv_6[2]
        qkv_6 = None
        x_34 = torch._C._nn.scaled_dot_product_attention(q_6, k_6, v_6, dropout_p=0.0)
        q_6 = k_6 = v_6 = None
        transpose_7 = x_34.transpose(1, 2)
        x_34 = None
        x_35 = transpose_7.reshape(1, 192, 1280)
        transpose_7 = None
        x_36 = torch._C._nn.linear(
            x_35,
            l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_proj_parameters_bias_,
        )
        x_35 = l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_proj_parameters_bias_ = (None)
        x_37 = torch.nn.functional.dropout(x_36, 0.0, False, False)
        x_36 = None
        x_38 = output_5 + x_37
        output_5 = x_37 = None
        layer_norm_13 = torch.nn.functional.layer_norm(
            x_38,
            (1280,),
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
        output_6 = x_38 + input_35
        x_38 = input_35 = None
        layer_norm_14 = torch.nn.functional.layer_norm(
            output_6,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_bias_ = (None)
        linear_28 = torch._C._nn.linear(
            layer_norm_14,
            l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_14 = l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_14 = linear_28.reshape(1, 192, 3, 16, 80)
        linear_28 = None
        qkv_7 = reshape_14.permute(2, 0, 3, 1, 4)
        reshape_14 = None
        q_7 = qkv_7[0]
        k_7 = qkv_7[1]
        v_7 = qkv_7[2]
        qkv_7 = None
        x_39 = torch._C._nn.scaled_dot_product_attention(q_7, k_7, v_7, dropout_p=0.0)
        q_7 = k_7 = v_7 = None
        transpose_8 = x_39.transpose(1, 2)
        x_39 = None
        x_40 = transpose_8.reshape(1, 192, 1280)
        transpose_8 = None
        x_41 = torch._C._nn.linear(
            x_40,
            l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_proj_parameters_bias_,
        )
        x_40 = l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_proj_parameters_bias_ = (None)
        x_42 = torch.nn.functional.dropout(x_41, 0.0, False, False)
        x_41 = None
        x_43 = output_6 + x_42
        output_6 = x_42 = None
        layer_norm_15 = torch.nn.functional.layer_norm(
            x_43,
            (1280,),
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
        output_7 = x_43 + input_40
        x_43 = input_40 = None
        layer_norm_16 = torch.nn.functional.layer_norm(
            output_7,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_bias_ = (None)
        linear_32 = torch._C._nn.linear(
            layer_norm_16,
            l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_16 = l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_16 = linear_32.reshape(1, 192, 3, 16, 80)
        linear_32 = None
        qkv_8 = reshape_16.permute(2, 0, 3, 1, 4)
        reshape_16 = None
        q_8 = qkv_8[0]
        k_8 = qkv_8[1]
        v_8 = qkv_8[2]
        qkv_8 = None
        x_44 = torch._C._nn.scaled_dot_product_attention(q_8, k_8, v_8, dropout_p=0.0)
        q_8 = k_8 = v_8 = None
        transpose_9 = x_44.transpose(1, 2)
        x_44 = None
        x_45 = transpose_9.reshape(1, 192, 1280)
        transpose_9 = None
        x_46 = torch._C._nn.linear(
            x_45,
            l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_proj_parameters_bias_,
        )
        x_45 = l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_proj_parameters_bias_ = (None)
        x_47 = torch.nn.functional.dropout(x_46, 0.0, False, False)
        x_46 = None
        x_48 = output_7 + x_47
        output_7 = x_47 = None
        layer_norm_17 = torch.nn.functional.layer_norm(
            x_48,
            (1280,),
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
        output_8 = x_48 + input_45
        x_48 = input_45 = None
        layer_norm_18 = torch.nn.functional.layer_norm(
            output_8,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_bias_ = (None)
        linear_36 = torch._C._nn.linear(
            layer_norm_18,
            l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_18 = l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_18 = linear_36.reshape(1, 192, 3, 16, 80)
        linear_36 = None
        qkv_9 = reshape_18.permute(2, 0, 3, 1, 4)
        reshape_18 = None
        q_9 = qkv_9[0]
        k_9 = qkv_9[1]
        v_9 = qkv_9[2]
        qkv_9 = None
        x_49 = torch._C._nn.scaled_dot_product_attention(q_9, k_9, v_9, dropout_p=0.0)
        q_9 = k_9 = v_9 = None
        transpose_10 = x_49.transpose(1, 2)
        x_49 = None
        x_50 = transpose_10.reshape(1, 192, 1280)
        transpose_10 = None
        x_51 = torch._C._nn.linear(
            x_50,
            l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_proj_parameters_bias_,
        )
        x_50 = l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_proj_parameters_bias_ = (None)
        x_52 = torch.nn.functional.dropout(x_51, 0.0, False, False)
        x_51 = None
        x_53 = output_8 + x_52
        output_8 = x_52 = None
        layer_norm_19 = torch.nn.functional.layer_norm(
            x_53,
            (1280,),
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
        output_9 = x_53 + input_50
        x_53 = input_50 = None
        layer_norm_20 = torch.nn.functional.layer_norm(
            output_9,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_bias_ = (None)
        linear_40 = torch._C._nn.linear(
            layer_norm_20,
            l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_20 = l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_20 = linear_40.reshape(1, 192, 3, 16, 80)
        linear_40 = None
        qkv_10 = reshape_20.permute(2, 0, 3, 1, 4)
        reshape_20 = None
        q_10 = qkv_10[0]
        k_10 = qkv_10[1]
        v_10 = qkv_10[2]
        qkv_10 = None
        x_54 = torch._C._nn.scaled_dot_product_attention(
            q_10, k_10, v_10, dropout_p=0.0
        )
        q_10 = k_10 = v_10 = None
        transpose_11 = x_54.transpose(1, 2)
        x_54 = None
        x_55 = transpose_11.reshape(1, 192, 1280)
        transpose_11 = None
        x_56 = torch._C._nn.linear(
            x_55,
            l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_proj_parameters_bias_,
        )
        x_55 = l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_proj_parameters_bias_ = (None)
        x_57 = torch.nn.functional.dropout(x_56, 0.0, False, False)
        x_56 = None
        x_58 = output_9 + x_57
        output_9 = x_57 = None
        layer_norm_21 = torch.nn.functional.layer_norm(
            x_58,
            (1280,),
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
        output_10 = x_58 + input_55
        x_58 = input_55 = None
        layer_norm_22 = torch.nn.functional.layer_norm(
            output_10,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_bias_ = (None)
        linear_44 = torch._C._nn.linear(
            layer_norm_22,
            l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_22 = l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_22 = linear_44.reshape(1, 192, 3, 16, 80)
        linear_44 = None
        qkv_11 = reshape_22.permute(2, 0, 3, 1, 4)
        reshape_22 = None
        q_11 = qkv_11[0]
        k_11 = qkv_11[1]
        v_11 = qkv_11[2]
        qkv_11 = None
        x_59 = torch._C._nn.scaled_dot_product_attention(
            q_11, k_11, v_11, dropout_p=0.0
        )
        q_11 = k_11 = v_11 = None
        transpose_12 = x_59.transpose(1, 2)
        x_59 = None
        x_60 = transpose_12.reshape(1, 192, 1280)
        transpose_12 = None
        x_61 = torch._C._nn.linear(
            x_60,
            l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_proj_parameters_bias_,
        )
        x_60 = l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_proj_parameters_bias_ = (None)
        x_62 = torch.nn.functional.dropout(x_61, 0.0, False, False)
        x_61 = None
        x_63 = output_10 + x_62
        output_10 = x_62 = None
        layer_norm_23 = torch.nn.functional.layer_norm(
            x_63,
            (1280,),
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
        output_11 = x_63 + input_60
        x_63 = input_60 = None
        layer_norm_24 = torch.nn.functional.layer_norm(
            output_11,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_12_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_12_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_12_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_12_modules_ln1_parameters_bias_ = (None)
        linear_48 = torch._C._nn.linear(
            layer_norm_24,
            l_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_24 = l_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_24 = linear_48.reshape(1, 192, 3, 16, 80)
        linear_48 = None
        qkv_12 = reshape_24.permute(2, 0, 3, 1, 4)
        reshape_24 = None
        q_12 = qkv_12[0]
        k_12 = qkv_12[1]
        v_12 = qkv_12[2]
        qkv_12 = None
        x_64 = torch._C._nn.scaled_dot_product_attention(
            q_12, k_12, v_12, dropout_p=0.0
        )
        q_12 = k_12 = v_12 = None
        transpose_13 = x_64.transpose(1, 2)
        x_64 = None
        x_65 = transpose_13.reshape(1, 192, 1280)
        transpose_13 = None
        x_66 = torch._C._nn.linear(
            x_65,
            l_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_proj_parameters_bias_,
        )
        x_65 = l_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_proj_parameters_bias_ = (None)
        x_67 = torch.nn.functional.dropout(x_66, 0.0, False, False)
        x_66 = None
        x_68 = output_11 + x_67
        output_11 = x_67 = None
        layer_norm_25 = torch.nn.functional.layer_norm(
            x_68,
            (1280,),
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
        output_12 = x_68 + input_65
        x_68 = input_65 = None
        layer_norm_26 = torch.nn.functional.layer_norm(
            output_12,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_13_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_13_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_13_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_13_modules_ln1_parameters_bias_ = (None)
        linear_52 = torch._C._nn.linear(
            layer_norm_26,
            l_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_26 = l_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_26 = linear_52.reshape(1, 192, 3, 16, 80)
        linear_52 = None
        qkv_13 = reshape_26.permute(2, 0, 3, 1, 4)
        reshape_26 = None
        q_13 = qkv_13[0]
        k_13 = qkv_13[1]
        v_13 = qkv_13[2]
        qkv_13 = None
        x_69 = torch._C._nn.scaled_dot_product_attention(
            q_13, k_13, v_13, dropout_p=0.0
        )
        q_13 = k_13 = v_13 = None
        transpose_14 = x_69.transpose(1, 2)
        x_69 = None
        x_70 = transpose_14.reshape(1, 192, 1280)
        transpose_14 = None
        x_71 = torch._C._nn.linear(
            x_70,
            l_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_proj_parameters_bias_,
        )
        x_70 = l_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_proj_parameters_bias_ = (None)
        x_72 = torch.nn.functional.dropout(x_71, 0.0, False, False)
        x_71 = None
        x_73 = output_12 + x_72
        output_12 = x_72 = None
        layer_norm_27 = torch.nn.functional.layer_norm(
            x_73,
            (1280,),
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
        output_13 = x_73 + input_70
        x_73 = input_70 = None
        layer_norm_28 = torch.nn.functional.layer_norm(
            output_13,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_14_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_14_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_14_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_14_modules_ln1_parameters_bias_ = (None)
        linear_56 = torch._C._nn.linear(
            layer_norm_28,
            l_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_28 = l_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_28 = linear_56.reshape(1, 192, 3, 16, 80)
        linear_56 = None
        qkv_14 = reshape_28.permute(2, 0, 3, 1, 4)
        reshape_28 = None
        q_14 = qkv_14[0]
        k_14 = qkv_14[1]
        v_14 = qkv_14[2]
        qkv_14 = None
        x_74 = torch._C._nn.scaled_dot_product_attention(
            q_14, k_14, v_14, dropout_p=0.0
        )
        q_14 = k_14 = v_14 = None
        transpose_15 = x_74.transpose(1, 2)
        x_74 = None
        x_75 = transpose_15.reshape(1, 192, 1280)
        transpose_15 = None
        x_76 = torch._C._nn.linear(
            x_75,
            l_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_proj_parameters_bias_,
        )
        x_75 = l_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_proj_parameters_bias_ = (None)
        x_77 = torch.nn.functional.dropout(x_76, 0.0, False, False)
        x_76 = None
        x_78 = output_13 + x_77
        output_13 = x_77 = None
        layer_norm_29 = torch.nn.functional.layer_norm(
            x_78,
            (1280,),
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
        output_14 = x_78 + input_75
        x_78 = input_75 = None
        layer_norm_30 = torch.nn.functional.layer_norm(
            output_14,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_15_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_15_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_15_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_15_modules_ln1_parameters_bias_ = (None)
        linear_60 = torch._C._nn.linear(
            layer_norm_30,
            l_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_30 = l_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_30 = linear_60.reshape(1, 192, 3, 16, 80)
        linear_60 = None
        qkv_15 = reshape_30.permute(2, 0, 3, 1, 4)
        reshape_30 = None
        q_15 = qkv_15[0]
        k_15 = qkv_15[1]
        v_15 = qkv_15[2]
        qkv_15 = None
        x_79 = torch._C._nn.scaled_dot_product_attention(
            q_15, k_15, v_15, dropout_p=0.0
        )
        q_15 = k_15 = v_15 = None
        transpose_16 = x_79.transpose(1, 2)
        x_79 = None
        x_80 = transpose_16.reshape(1, 192, 1280)
        transpose_16 = None
        x_81 = torch._C._nn.linear(
            x_80,
            l_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_proj_parameters_bias_,
        )
        x_80 = l_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_proj_parameters_bias_ = (None)
        x_82 = torch.nn.functional.dropout(x_81, 0.0, False, False)
        x_81 = None
        x_83 = output_14 + x_82
        output_14 = x_82 = None
        layer_norm_31 = torch.nn.functional.layer_norm(
            x_83,
            (1280,),
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
        output_15 = x_83 + input_80
        x_83 = input_80 = None
        layer_norm_32 = torch.nn.functional.layer_norm(
            output_15,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_16_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_16_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_16_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_16_modules_ln1_parameters_bias_ = (None)
        linear_64 = torch._C._nn.linear(
            layer_norm_32,
            l_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_32 = l_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_32 = linear_64.reshape(1, 192, 3, 16, 80)
        linear_64 = None
        qkv_16 = reshape_32.permute(2, 0, 3, 1, 4)
        reshape_32 = None
        q_16 = qkv_16[0]
        k_16 = qkv_16[1]
        v_16 = qkv_16[2]
        qkv_16 = None
        x_84 = torch._C._nn.scaled_dot_product_attention(
            q_16, k_16, v_16, dropout_p=0.0
        )
        q_16 = k_16 = v_16 = None
        transpose_17 = x_84.transpose(1, 2)
        x_84 = None
        x_85 = transpose_17.reshape(1, 192, 1280)
        transpose_17 = None
        x_86 = torch._C._nn.linear(
            x_85,
            l_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_proj_parameters_bias_,
        )
        x_85 = l_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_proj_parameters_bias_ = (None)
        x_87 = torch.nn.functional.dropout(x_86, 0.0, False, False)
        x_86 = None
        x_88 = output_15 + x_87
        output_15 = x_87 = None
        layer_norm_33 = torch.nn.functional.layer_norm(
            x_88,
            (1280,),
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
        output_16 = x_88 + input_85
        x_88 = input_85 = None
        layer_norm_34 = torch.nn.functional.layer_norm(
            output_16,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_17_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_17_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_17_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_17_modules_ln1_parameters_bias_ = (None)
        linear_68 = torch._C._nn.linear(
            layer_norm_34,
            l_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_34 = l_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_34 = linear_68.reshape(1, 192, 3, 16, 80)
        linear_68 = None
        qkv_17 = reshape_34.permute(2, 0, 3, 1, 4)
        reshape_34 = None
        q_17 = qkv_17[0]
        k_17 = qkv_17[1]
        v_17 = qkv_17[2]
        qkv_17 = None
        x_89 = torch._C._nn.scaled_dot_product_attention(
            q_17, k_17, v_17, dropout_p=0.0
        )
        q_17 = k_17 = v_17 = None
        transpose_18 = x_89.transpose(1, 2)
        x_89 = None
        x_90 = transpose_18.reshape(1, 192, 1280)
        transpose_18 = None
        x_91 = torch._C._nn.linear(
            x_90,
            l_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_proj_parameters_bias_,
        )
        x_90 = l_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_proj_parameters_bias_ = (None)
        x_92 = torch.nn.functional.dropout(x_91, 0.0, False, False)
        x_91 = None
        x_93 = output_16 + x_92
        output_16 = x_92 = None
        layer_norm_35 = torch.nn.functional.layer_norm(
            x_93,
            (1280,),
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
        output_17 = x_93 + input_90
        x_93 = input_90 = None
        layer_norm_36 = torch.nn.functional.layer_norm(
            output_17,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_18_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_18_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_18_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_18_modules_ln1_parameters_bias_ = (None)
        linear_72 = torch._C._nn.linear(
            layer_norm_36,
            l_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_36 = l_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_36 = linear_72.reshape(1, 192, 3, 16, 80)
        linear_72 = None
        qkv_18 = reshape_36.permute(2, 0, 3, 1, 4)
        reshape_36 = None
        q_18 = qkv_18[0]
        k_18 = qkv_18[1]
        v_18 = qkv_18[2]
        qkv_18 = None
        x_94 = torch._C._nn.scaled_dot_product_attention(
            q_18, k_18, v_18, dropout_p=0.0
        )
        q_18 = k_18 = v_18 = None
        transpose_19 = x_94.transpose(1, 2)
        x_94 = None
        x_95 = transpose_19.reshape(1, 192, 1280)
        transpose_19 = None
        x_96 = torch._C._nn.linear(
            x_95,
            l_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_proj_parameters_bias_,
        )
        x_95 = l_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_proj_parameters_bias_ = (None)
        x_97 = torch.nn.functional.dropout(x_96, 0.0, False, False)
        x_96 = None
        x_98 = output_17 + x_97
        output_17 = x_97 = None
        layer_norm_37 = torch.nn.functional.layer_norm(
            x_98,
            (1280,),
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
        output_18 = x_98 + input_95
        x_98 = input_95 = None
        layer_norm_38 = torch.nn.functional.layer_norm(
            output_18,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_19_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_19_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_19_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_19_modules_ln1_parameters_bias_ = (None)
        linear_76 = torch._C._nn.linear(
            layer_norm_38,
            l_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_38 = l_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_38 = linear_76.reshape(1, 192, 3, 16, 80)
        linear_76 = None
        qkv_19 = reshape_38.permute(2, 0, 3, 1, 4)
        reshape_38 = None
        q_19 = qkv_19[0]
        k_19 = qkv_19[1]
        v_19 = qkv_19[2]
        qkv_19 = None
        x_99 = torch._C._nn.scaled_dot_product_attention(
            q_19, k_19, v_19, dropout_p=0.0
        )
        q_19 = k_19 = v_19 = None
        transpose_20 = x_99.transpose(1, 2)
        x_99 = None
        x_100 = transpose_20.reshape(1, 192, 1280)
        transpose_20 = None
        x_101 = torch._C._nn.linear(
            x_100,
            l_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_proj_parameters_bias_,
        )
        x_100 = l_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_proj_parameters_bias_ = (None)
        x_102 = torch.nn.functional.dropout(x_101, 0.0, False, False)
        x_101 = None
        x_103 = output_18 + x_102
        output_18 = x_102 = None
        layer_norm_39 = torch.nn.functional.layer_norm(
            x_103,
            (1280,),
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
        output_19 = x_103 + input_100
        x_103 = input_100 = None
        layer_norm_40 = torch.nn.functional.layer_norm(
            output_19,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_20_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_20_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_20_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_20_modules_ln1_parameters_bias_ = (None)
        linear_80 = torch._C._nn.linear(
            layer_norm_40,
            l_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_40 = l_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_40 = linear_80.reshape(1, 192, 3, 16, 80)
        linear_80 = None
        qkv_20 = reshape_40.permute(2, 0, 3, 1, 4)
        reshape_40 = None
        q_20 = qkv_20[0]
        k_20 = qkv_20[1]
        v_20 = qkv_20[2]
        qkv_20 = None
        x_104 = torch._C._nn.scaled_dot_product_attention(
            q_20, k_20, v_20, dropout_p=0.0
        )
        q_20 = k_20 = v_20 = None
        transpose_21 = x_104.transpose(1, 2)
        x_104 = None
        x_105 = transpose_21.reshape(1, 192, 1280)
        transpose_21 = None
        x_106 = torch._C._nn.linear(
            x_105,
            l_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_proj_parameters_bias_,
        )
        x_105 = l_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_proj_parameters_bias_ = (None)
        x_107 = torch.nn.functional.dropout(x_106, 0.0, False, False)
        x_106 = None
        x_108 = output_19 + x_107
        output_19 = x_107 = None
        layer_norm_41 = torch.nn.functional.layer_norm(
            x_108,
            (1280,),
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
        output_20 = x_108 + input_105
        x_108 = input_105 = None
        layer_norm_42 = torch.nn.functional.layer_norm(
            output_20,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_21_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_21_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_21_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_21_modules_ln1_parameters_bias_ = (None)
        linear_84 = torch._C._nn.linear(
            layer_norm_42,
            l_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_42 = l_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_42 = linear_84.reshape(1, 192, 3, 16, 80)
        linear_84 = None
        qkv_21 = reshape_42.permute(2, 0, 3, 1, 4)
        reshape_42 = None
        q_21 = qkv_21[0]
        k_21 = qkv_21[1]
        v_21 = qkv_21[2]
        qkv_21 = None
        x_109 = torch._C._nn.scaled_dot_product_attention(
            q_21, k_21, v_21, dropout_p=0.0
        )
        q_21 = k_21 = v_21 = None
        transpose_22 = x_109.transpose(1, 2)
        x_109 = None
        x_110 = transpose_22.reshape(1, 192, 1280)
        transpose_22 = None
        x_111 = torch._C._nn.linear(
            x_110,
            l_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_proj_parameters_bias_,
        )
        x_110 = l_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_proj_parameters_bias_ = (None)
        x_112 = torch.nn.functional.dropout(x_111, 0.0, False, False)
        x_111 = None
        x_113 = output_20 + x_112
        output_20 = x_112 = None
        layer_norm_43 = torch.nn.functional.layer_norm(
            x_113,
            (1280,),
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
        output_21 = x_113 + input_110
        x_113 = input_110 = None
        layer_norm_44 = torch.nn.functional.layer_norm(
            output_21,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_22_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_22_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_22_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_22_modules_ln1_parameters_bias_ = (None)
        linear_88 = torch._C._nn.linear(
            layer_norm_44,
            l_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_44 = l_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_44 = linear_88.reshape(1, 192, 3, 16, 80)
        linear_88 = None
        qkv_22 = reshape_44.permute(2, 0, 3, 1, 4)
        reshape_44 = None
        q_22 = qkv_22[0]
        k_22 = qkv_22[1]
        v_22 = qkv_22[2]
        qkv_22 = None
        x_114 = torch._C._nn.scaled_dot_product_attention(
            q_22, k_22, v_22, dropout_p=0.0
        )
        q_22 = k_22 = v_22 = None
        transpose_23 = x_114.transpose(1, 2)
        x_114 = None
        x_115 = transpose_23.reshape(1, 192, 1280)
        transpose_23 = None
        x_116 = torch._C._nn.linear(
            x_115,
            l_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_proj_parameters_bias_,
        )
        x_115 = l_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_proj_parameters_bias_ = (None)
        x_117 = torch.nn.functional.dropout(x_116, 0.0, False, False)
        x_116 = None
        x_118 = output_21 + x_117
        output_21 = x_117 = None
        layer_norm_45 = torch.nn.functional.layer_norm(
            x_118,
            (1280,),
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
        output_22 = x_118 + input_115
        x_118 = input_115 = None
        layer_norm_46 = torch.nn.functional.layer_norm(
            output_22,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_23_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_23_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_23_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_23_modules_ln1_parameters_bias_ = (None)
        linear_92 = torch._C._nn.linear(
            layer_norm_46,
            l_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_46 = l_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_46 = linear_92.reshape(1, 192, 3, 16, 80)
        linear_92 = None
        qkv_23 = reshape_46.permute(2, 0, 3, 1, 4)
        reshape_46 = None
        q_23 = qkv_23[0]
        k_23 = qkv_23[1]
        v_23 = qkv_23[2]
        qkv_23 = None
        x_119 = torch._C._nn.scaled_dot_product_attention(
            q_23, k_23, v_23, dropout_p=0.0
        )
        q_23 = k_23 = v_23 = None
        transpose_24 = x_119.transpose(1, 2)
        x_119 = None
        x_120 = transpose_24.reshape(1, 192, 1280)
        transpose_24 = None
        x_121 = torch._C._nn.linear(
            x_120,
            l_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_proj_parameters_bias_,
        )
        x_120 = l_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_proj_parameters_bias_ = (None)
        x_122 = torch.nn.functional.dropout(x_121, 0.0, False, False)
        x_121 = None
        x_123 = output_22 + x_122
        output_22 = x_122 = None
        layer_norm_47 = torch.nn.functional.layer_norm(
            x_123,
            (1280,),
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
        output_23 = x_123 + input_120
        x_123 = input_120 = None
        layer_norm_48 = torch.nn.functional.layer_norm(
            output_23,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_24_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_24_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_24_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_24_modules_ln1_parameters_bias_ = (None)
        linear_96 = torch._C._nn.linear(
            layer_norm_48,
            l_self_modules_backbone_modules_layers_modules_24_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_24_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_48 = l_self_modules_backbone_modules_layers_modules_24_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_24_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_48 = linear_96.reshape(1, 192, 3, 16, 80)
        linear_96 = None
        qkv_24 = reshape_48.permute(2, 0, 3, 1, 4)
        reshape_48 = None
        q_24 = qkv_24[0]
        k_24 = qkv_24[1]
        v_24 = qkv_24[2]
        qkv_24 = None
        x_124 = torch._C._nn.scaled_dot_product_attention(
            q_24, k_24, v_24, dropout_p=0.0
        )
        q_24 = k_24 = v_24 = None
        transpose_25 = x_124.transpose(1, 2)
        x_124 = None
        x_125 = transpose_25.reshape(1, 192, 1280)
        transpose_25 = None
        x_126 = torch._C._nn.linear(
            x_125,
            l_self_modules_backbone_modules_layers_modules_24_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_24_modules_attn_modules_proj_parameters_bias_,
        )
        x_125 = l_self_modules_backbone_modules_layers_modules_24_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_24_modules_attn_modules_proj_parameters_bias_ = (None)
        x_127 = torch.nn.functional.dropout(x_126, 0.0, False, False)
        x_126 = None
        x_128 = output_23 + x_127
        output_23 = x_127 = None
        layer_norm_49 = torch.nn.functional.layer_norm(
            x_128,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_24_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_24_modules_ln2_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_24_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_24_modules_ln2_parameters_bias_ = (None)
        input_121 = torch._C._nn.linear(
            layer_norm_49,
            l_self_modules_backbone_modules_layers_modules_24_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_24_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_49 = l_self_modules_backbone_modules_layers_modules_24_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_24_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_122 = torch._C._nn.gelu(input_121, approximate="none")
        input_121 = None
        input_123 = torch.nn.functional.dropout(input_122, 0.0, False, False)
        input_122 = None
        input_124 = torch._C._nn.linear(
            input_123,
            l_self_modules_backbone_modules_layers_modules_24_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_24_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_123 = l_self_modules_backbone_modules_layers_modules_24_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_24_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_125 = torch.nn.functional.dropout(input_124, 0.0, False, False)
        input_124 = None
        output_24 = x_128 + input_125
        x_128 = input_125 = None
        layer_norm_50 = torch.nn.functional.layer_norm(
            output_24,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_25_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_25_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_25_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_25_modules_ln1_parameters_bias_ = (None)
        linear_100 = torch._C._nn.linear(
            layer_norm_50,
            l_self_modules_backbone_modules_layers_modules_25_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_25_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_50 = l_self_modules_backbone_modules_layers_modules_25_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_25_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_50 = linear_100.reshape(1, 192, 3, 16, 80)
        linear_100 = None
        qkv_25 = reshape_50.permute(2, 0, 3, 1, 4)
        reshape_50 = None
        q_25 = qkv_25[0]
        k_25 = qkv_25[1]
        v_25 = qkv_25[2]
        qkv_25 = None
        x_129 = torch._C._nn.scaled_dot_product_attention(
            q_25, k_25, v_25, dropout_p=0.0
        )
        q_25 = k_25 = v_25 = None
        transpose_26 = x_129.transpose(1, 2)
        x_129 = None
        x_130 = transpose_26.reshape(1, 192, 1280)
        transpose_26 = None
        x_131 = torch._C._nn.linear(
            x_130,
            l_self_modules_backbone_modules_layers_modules_25_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_25_modules_attn_modules_proj_parameters_bias_,
        )
        x_130 = l_self_modules_backbone_modules_layers_modules_25_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_25_modules_attn_modules_proj_parameters_bias_ = (None)
        x_132 = torch.nn.functional.dropout(x_131, 0.0, False, False)
        x_131 = None
        x_133 = output_24 + x_132
        output_24 = x_132 = None
        layer_norm_51 = torch.nn.functional.layer_norm(
            x_133,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_25_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_25_modules_ln2_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_25_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_25_modules_ln2_parameters_bias_ = (None)
        input_126 = torch._C._nn.linear(
            layer_norm_51,
            l_self_modules_backbone_modules_layers_modules_25_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_25_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_51 = l_self_modules_backbone_modules_layers_modules_25_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_25_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_127 = torch._C._nn.gelu(input_126, approximate="none")
        input_126 = None
        input_128 = torch.nn.functional.dropout(input_127, 0.0, False, False)
        input_127 = None
        input_129 = torch._C._nn.linear(
            input_128,
            l_self_modules_backbone_modules_layers_modules_25_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_25_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_128 = l_self_modules_backbone_modules_layers_modules_25_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_25_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_130 = torch.nn.functional.dropout(input_129, 0.0, False, False)
        input_129 = None
        output_25 = x_133 + input_130
        x_133 = input_130 = None
        layer_norm_52 = torch.nn.functional.layer_norm(
            output_25,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_26_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_26_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_26_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_26_modules_ln1_parameters_bias_ = (None)
        linear_104 = torch._C._nn.linear(
            layer_norm_52,
            l_self_modules_backbone_modules_layers_modules_26_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_26_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_52 = l_self_modules_backbone_modules_layers_modules_26_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_26_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_52 = linear_104.reshape(1, 192, 3, 16, 80)
        linear_104 = None
        qkv_26 = reshape_52.permute(2, 0, 3, 1, 4)
        reshape_52 = None
        q_26 = qkv_26[0]
        k_26 = qkv_26[1]
        v_26 = qkv_26[2]
        qkv_26 = None
        x_134 = torch._C._nn.scaled_dot_product_attention(
            q_26, k_26, v_26, dropout_p=0.0
        )
        q_26 = k_26 = v_26 = None
        transpose_27 = x_134.transpose(1, 2)
        x_134 = None
        x_135 = transpose_27.reshape(1, 192, 1280)
        transpose_27 = None
        x_136 = torch._C._nn.linear(
            x_135,
            l_self_modules_backbone_modules_layers_modules_26_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_26_modules_attn_modules_proj_parameters_bias_,
        )
        x_135 = l_self_modules_backbone_modules_layers_modules_26_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_26_modules_attn_modules_proj_parameters_bias_ = (None)
        x_137 = torch.nn.functional.dropout(x_136, 0.0, False, False)
        x_136 = None
        x_138 = output_25 + x_137
        output_25 = x_137 = None
        layer_norm_53 = torch.nn.functional.layer_norm(
            x_138,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_26_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_26_modules_ln2_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_26_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_26_modules_ln2_parameters_bias_ = (None)
        input_131 = torch._C._nn.linear(
            layer_norm_53,
            l_self_modules_backbone_modules_layers_modules_26_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_26_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_53 = l_self_modules_backbone_modules_layers_modules_26_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_26_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_132 = torch._C._nn.gelu(input_131, approximate="none")
        input_131 = None
        input_133 = torch.nn.functional.dropout(input_132, 0.0, False, False)
        input_132 = None
        input_134 = torch._C._nn.linear(
            input_133,
            l_self_modules_backbone_modules_layers_modules_26_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_26_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_133 = l_self_modules_backbone_modules_layers_modules_26_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_26_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_135 = torch.nn.functional.dropout(input_134, 0.0, False, False)
        input_134 = None
        output_26 = x_138 + input_135
        x_138 = input_135 = None
        layer_norm_54 = torch.nn.functional.layer_norm(
            output_26,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_27_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_27_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_27_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_27_modules_ln1_parameters_bias_ = (None)
        linear_108 = torch._C._nn.linear(
            layer_norm_54,
            l_self_modules_backbone_modules_layers_modules_27_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_27_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_54 = l_self_modules_backbone_modules_layers_modules_27_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_27_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_54 = linear_108.reshape(1, 192, 3, 16, 80)
        linear_108 = None
        qkv_27 = reshape_54.permute(2, 0, 3, 1, 4)
        reshape_54 = None
        q_27 = qkv_27[0]
        k_27 = qkv_27[1]
        v_27 = qkv_27[2]
        qkv_27 = None
        x_139 = torch._C._nn.scaled_dot_product_attention(
            q_27, k_27, v_27, dropout_p=0.0
        )
        q_27 = k_27 = v_27 = None
        transpose_28 = x_139.transpose(1, 2)
        x_139 = None
        x_140 = transpose_28.reshape(1, 192, 1280)
        transpose_28 = None
        x_141 = torch._C._nn.linear(
            x_140,
            l_self_modules_backbone_modules_layers_modules_27_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_27_modules_attn_modules_proj_parameters_bias_,
        )
        x_140 = l_self_modules_backbone_modules_layers_modules_27_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_27_modules_attn_modules_proj_parameters_bias_ = (None)
        x_142 = torch.nn.functional.dropout(x_141, 0.0, False, False)
        x_141 = None
        x_143 = output_26 + x_142
        output_26 = x_142 = None
        layer_norm_55 = torch.nn.functional.layer_norm(
            x_143,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_27_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_27_modules_ln2_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_27_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_27_modules_ln2_parameters_bias_ = (None)
        input_136 = torch._C._nn.linear(
            layer_norm_55,
            l_self_modules_backbone_modules_layers_modules_27_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_27_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_55 = l_self_modules_backbone_modules_layers_modules_27_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_27_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_137 = torch._C._nn.gelu(input_136, approximate="none")
        input_136 = None
        input_138 = torch.nn.functional.dropout(input_137, 0.0, False, False)
        input_137 = None
        input_139 = torch._C._nn.linear(
            input_138,
            l_self_modules_backbone_modules_layers_modules_27_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_27_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_138 = l_self_modules_backbone_modules_layers_modules_27_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_27_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_140 = torch.nn.functional.dropout(input_139, 0.0, False, False)
        input_139 = None
        output_27 = x_143 + input_140
        x_143 = input_140 = None
        layer_norm_56 = torch.nn.functional.layer_norm(
            output_27,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_28_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_28_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_28_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_28_modules_ln1_parameters_bias_ = (None)
        linear_112 = torch._C._nn.linear(
            layer_norm_56,
            l_self_modules_backbone_modules_layers_modules_28_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_28_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_56 = l_self_modules_backbone_modules_layers_modules_28_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_28_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_56 = linear_112.reshape(1, 192, 3, 16, 80)
        linear_112 = None
        qkv_28 = reshape_56.permute(2, 0, 3, 1, 4)
        reshape_56 = None
        q_28 = qkv_28[0]
        k_28 = qkv_28[1]
        v_28 = qkv_28[2]
        qkv_28 = None
        x_144 = torch._C._nn.scaled_dot_product_attention(
            q_28, k_28, v_28, dropout_p=0.0
        )
        q_28 = k_28 = v_28 = None
        transpose_29 = x_144.transpose(1, 2)
        x_144 = None
        x_145 = transpose_29.reshape(1, 192, 1280)
        transpose_29 = None
        x_146 = torch._C._nn.linear(
            x_145,
            l_self_modules_backbone_modules_layers_modules_28_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_28_modules_attn_modules_proj_parameters_bias_,
        )
        x_145 = l_self_modules_backbone_modules_layers_modules_28_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_28_modules_attn_modules_proj_parameters_bias_ = (None)
        x_147 = torch.nn.functional.dropout(x_146, 0.0, False, False)
        x_146 = None
        x_148 = output_27 + x_147
        output_27 = x_147 = None
        layer_norm_57 = torch.nn.functional.layer_norm(
            x_148,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_28_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_28_modules_ln2_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_28_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_28_modules_ln2_parameters_bias_ = (None)
        input_141 = torch._C._nn.linear(
            layer_norm_57,
            l_self_modules_backbone_modules_layers_modules_28_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_28_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_57 = l_self_modules_backbone_modules_layers_modules_28_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_28_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_142 = torch._C._nn.gelu(input_141, approximate="none")
        input_141 = None
        input_143 = torch.nn.functional.dropout(input_142, 0.0, False, False)
        input_142 = None
        input_144 = torch._C._nn.linear(
            input_143,
            l_self_modules_backbone_modules_layers_modules_28_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_28_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_143 = l_self_modules_backbone_modules_layers_modules_28_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_28_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_145 = torch.nn.functional.dropout(input_144, 0.0, False, False)
        input_144 = None
        output_28 = x_148 + input_145
        x_148 = input_145 = None
        layer_norm_58 = torch.nn.functional.layer_norm(
            output_28,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_29_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_29_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_29_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_29_modules_ln1_parameters_bias_ = (None)
        linear_116 = torch._C._nn.linear(
            layer_norm_58,
            l_self_modules_backbone_modules_layers_modules_29_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_29_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_58 = l_self_modules_backbone_modules_layers_modules_29_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_29_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_58 = linear_116.reshape(1, 192, 3, 16, 80)
        linear_116 = None
        qkv_29 = reshape_58.permute(2, 0, 3, 1, 4)
        reshape_58 = None
        q_29 = qkv_29[0]
        k_29 = qkv_29[1]
        v_29 = qkv_29[2]
        qkv_29 = None
        x_149 = torch._C._nn.scaled_dot_product_attention(
            q_29, k_29, v_29, dropout_p=0.0
        )
        q_29 = k_29 = v_29 = None
        transpose_30 = x_149.transpose(1, 2)
        x_149 = None
        x_150 = transpose_30.reshape(1, 192, 1280)
        transpose_30 = None
        x_151 = torch._C._nn.linear(
            x_150,
            l_self_modules_backbone_modules_layers_modules_29_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_29_modules_attn_modules_proj_parameters_bias_,
        )
        x_150 = l_self_modules_backbone_modules_layers_modules_29_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_29_modules_attn_modules_proj_parameters_bias_ = (None)
        x_152 = torch.nn.functional.dropout(x_151, 0.0, False, False)
        x_151 = None
        x_153 = output_28 + x_152
        output_28 = x_152 = None
        layer_norm_59 = torch.nn.functional.layer_norm(
            x_153,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_29_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_29_modules_ln2_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_29_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_29_modules_ln2_parameters_bias_ = (None)
        input_146 = torch._C._nn.linear(
            layer_norm_59,
            l_self_modules_backbone_modules_layers_modules_29_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_29_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_59 = l_self_modules_backbone_modules_layers_modules_29_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_29_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_147 = torch._C._nn.gelu(input_146, approximate="none")
        input_146 = None
        input_148 = torch.nn.functional.dropout(input_147, 0.0, False, False)
        input_147 = None
        input_149 = torch._C._nn.linear(
            input_148,
            l_self_modules_backbone_modules_layers_modules_29_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_29_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_148 = l_self_modules_backbone_modules_layers_modules_29_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_29_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_150 = torch.nn.functional.dropout(input_149, 0.0, False, False)
        input_149 = None
        output_29 = x_153 + input_150
        x_153 = input_150 = None
        layer_norm_60 = torch.nn.functional.layer_norm(
            output_29,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_30_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_30_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_30_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_30_modules_ln1_parameters_bias_ = (None)
        linear_120 = torch._C._nn.linear(
            layer_norm_60,
            l_self_modules_backbone_modules_layers_modules_30_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_30_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_60 = l_self_modules_backbone_modules_layers_modules_30_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_30_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_60 = linear_120.reshape(1, 192, 3, 16, 80)
        linear_120 = None
        qkv_30 = reshape_60.permute(2, 0, 3, 1, 4)
        reshape_60 = None
        q_30 = qkv_30[0]
        k_30 = qkv_30[1]
        v_30 = qkv_30[2]
        qkv_30 = None
        x_154 = torch._C._nn.scaled_dot_product_attention(
            q_30, k_30, v_30, dropout_p=0.0
        )
        q_30 = k_30 = v_30 = None
        transpose_31 = x_154.transpose(1, 2)
        x_154 = None
        x_155 = transpose_31.reshape(1, 192, 1280)
        transpose_31 = None
        x_156 = torch._C._nn.linear(
            x_155,
            l_self_modules_backbone_modules_layers_modules_30_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_30_modules_attn_modules_proj_parameters_bias_,
        )
        x_155 = l_self_modules_backbone_modules_layers_modules_30_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_30_modules_attn_modules_proj_parameters_bias_ = (None)
        x_157 = torch.nn.functional.dropout(x_156, 0.0, False, False)
        x_156 = None
        x_158 = output_29 + x_157
        output_29 = x_157 = None
        layer_norm_61 = torch.nn.functional.layer_norm(
            x_158,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_30_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_30_modules_ln2_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_30_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_30_modules_ln2_parameters_bias_ = (None)
        input_151 = torch._C._nn.linear(
            layer_norm_61,
            l_self_modules_backbone_modules_layers_modules_30_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_30_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_61 = l_self_modules_backbone_modules_layers_modules_30_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_30_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_152 = torch._C._nn.gelu(input_151, approximate="none")
        input_151 = None
        input_153 = torch.nn.functional.dropout(input_152, 0.0, False, False)
        input_152 = None
        input_154 = torch._C._nn.linear(
            input_153,
            l_self_modules_backbone_modules_layers_modules_30_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_30_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_153 = l_self_modules_backbone_modules_layers_modules_30_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_30_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_155 = torch.nn.functional.dropout(input_154, 0.0, False, False)
        input_154 = None
        output_30 = x_158 + input_155
        x_158 = input_155 = None
        layer_norm_62 = torch.nn.functional.layer_norm(
            output_30,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_31_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_31_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_31_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_31_modules_ln1_parameters_bias_ = (None)
        linear_124 = torch._C._nn.linear(
            layer_norm_62,
            l_self_modules_backbone_modules_layers_modules_31_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_31_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_62 = l_self_modules_backbone_modules_layers_modules_31_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_31_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_62 = linear_124.reshape(1, 192, 3, 16, 80)
        linear_124 = None
        qkv_31 = reshape_62.permute(2, 0, 3, 1, 4)
        reshape_62 = None
        q_31 = qkv_31[0]
        k_31 = qkv_31[1]
        v_31 = qkv_31[2]
        qkv_31 = None
        x_159 = torch._C._nn.scaled_dot_product_attention(
            q_31, k_31, v_31, dropout_p=0.0
        )
        q_31 = k_31 = v_31 = None
        transpose_32 = x_159.transpose(1, 2)
        x_159 = None
        x_160 = transpose_32.reshape(1, 192, 1280)
        transpose_32 = None
        x_161 = torch._C._nn.linear(
            x_160,
            l_self_modules_backbone_modules_layers_modules_31_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_31_modules_attn_modules_proj_parameters_bias_,
        )
        x_160 = l_self_modules_backbone_modules_layers_modules_31_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_31_modules_attn_modules_proj_parameters_bias_ = (None)
        x_162 = torch.nn.functional.dropout(x_161, 0.0, False, False)
        x_161 = None
        x_163 = output_30 + x_162
        output_30 = x_162 = None
        layer_norm_63 = torch.nn.functional.layer_norm(
            x_163,
            (1280,),
            l_self_modules_backbone_modules_layers_modules_31_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_31_modules_ln2_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_31_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_31_modules_ln2_parameters_bias_ = (None)
        input_156 = torch._C._nn.linear(
            layer_norm_63,
            l_self_modules_backbone_modules_layers_modules_31_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_31_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_63 = l_self_modules_backbone_modules_layers_modules_31_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_31_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_157 = torch._C._nn.gelu(input_156, approximate="none")
        input_156 = None
        input_158 = torch.nn.functional.dropout(input_157, 0.0, False, False)
        input_157 = None
        input_159 = torch._C._nn.linear(
            input_158,
            l_self_modules_backbone_modules_layers_modules_31_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_31_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_158 = l_self_modules_backbone_modules_layers_modules_31_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_31_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_160 = torch.nn.functional.dropout(input_159, 0.0, False, False)
        input_159 = None
        output_31 = x_163 + input_160
        x_163 = input_160 = None
        x_164 = torch.nn.functional.layer_norm(
            output_31,
            (1280,),
            l_self_modules_backbone_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_ln1_parameters_bias_,
            1e-06,
        )
        output_31 = (
            l_self_modules_backbone_modules_ln1_parameters_weight_
        ) = l_self_modules_backbone_modules_ln1_parameters_bias_ = None
        patch_token = x_164[(slice(None, None, None), slice(0, None, None))]
        x_164 = None
        reshape_64 = patch_token.reshape(1, 16, 12, -1)
        patch_token = None
        x_165 = reshape_64.permute(0, 3, 1, 2)
        reshape_64 = None
        input_161 = torch.conv_transpose2d(
            x_165,
            l_self_modules_head_modules_deconv_layers_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (0, 0),
            1,
            (1, 1),
        )
        x_165 = (
            l_self_modules_head_modules_deconv_layers_modules_0_parameters_weight_
        ) = None
        input_162 = torch.nn.functional.batch_norm(
            input_161,
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_mean_,
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_var_,
            l_self_modules_head_modules_deconv_layers_modules_1_parameters_weight_,
            l_self_modules_head_modules_deconv_layers_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_161 = (
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_var_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_1_parameters_weight_
        ) = l_self_modules_head_modules_deconv_layers_modules_1_parameters_bias_ = None
        input_163 = torch.nn.functional.relu(input_162, inplace=True)
        input_162 = None
        input_164 = torch.conv_transpose2d(
            input_163,
            l_self_modules_head_modules_deconv_layers_modules_3_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (0, 0),
            1,
            (1, 1),
        )
        input_163 = (
            l_self_modules_head_modules_deconv_layers_modules_3_parameters_weight_
        ) = None
        input_165 = torch.nn.functional.batch_norm(
            input_164,
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_mean_,
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_var_,
            l_self_modules_head_modules_deconv_layers_modules_4_parameters_weight_,
            l_self_modules_head_modules_deconv_layers_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_164 = (
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_mean_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_var_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_4_parameters_weight_
        ) = l_self_modules_head_modules_deconv_layers_modules_4_parameters_bias_ = None
        input_166 = torch.nn.functional.relu(input_165, inplace=True)
        input_165 = None
        x_166 = torch.conv2d(
            input_166,
            l_self_modules_head_modules_final_layer_parameters_weight_,
            l_self_modules_head_modules_final_layer_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_166 = (
            l_self_modules_head_modules_final_layer_parameters_weight_
        ) = l_self_modules_head_modules_final_layer_parameters_bias_ = None
        return (x_166,)
