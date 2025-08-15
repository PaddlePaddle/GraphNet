import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_inputs_: torch.Tensor,
        L_self_modules_backbone_modules_patch_embed_modules_projection_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_parameters_cls_token_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_parameters_pos_embed_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_12_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_12_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_12_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_12_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_norm_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_norm_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_norm_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_norm_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_norm_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_norm_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_norm_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_norm_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_mla_modules_channel_proj_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_mla_modules_channel_proj_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_mla_modules_channel_proj_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_mla_modules_channel_proj_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_mla_modules_channel_proj_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_mla_modules_channel_proj_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_mla_modules_channel_proj_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_mla_modules_channel_proj_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_mla_modules_channel_proj_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_mla_modules_channel_proj_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_mla_modules_channel_proj_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_mla_modules_channel_proj_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_mla_modules_channel_proj_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_mla_modules_channel_proj_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_mla_modules_channel_proj_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_mla_modules_channel_proj_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_mla_modules_channel_proj_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_mla_modules_channel_proj_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_mla_modules_channel_proj_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_mla_modules_channel_proj_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_mla_modules_feat_extract_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_mla_modules_feat_extract_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_mla_modules_feat_extract_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_mla_modules_feat_extract_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_mla_modules_feat_extract_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_mla_modules_feat_extract_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_mla_modules_feat_extract_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_mla_modules_feat_extract_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_mla_modules_feat_extract_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_mla_modules_feat_extract_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_mla_modules_feat_extract_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_mla_modules_feat_extract_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_mla_modules_feat_extract_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_mla_modules_feat_extract_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_mla_modules_feat_extract_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_mla_modules_feat_extract_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_mla_modules_feat_extract_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_mla_modules_feat_extract_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_mla_modules_feat_extract_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_mla_modules_feat_extract_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_up_convs_modules_0_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_up_convs_modules_0_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_up_convs_modules_0_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_up_convs_modules_0_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_up_convs_modules_0_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_up_convs_modules_0_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_up_convs_modules_0_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_up_convs_modules_0_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_up_convs_modules_0_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_up_convs_modules_0_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_up_convs_modules_1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_up_convs_modules_1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_up_convs_modules_1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_up_convs_modules_1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_up_convs_modules_1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_up_convs_modules_1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_up_convs_modules_1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_up_convs_modules_1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_up_convs_modules_1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_up_convs_modules_1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_up_convs_modules_2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_up_convs_modules_2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_up_convs_modules_2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_up_convs_modules_2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_up_convs_modules_2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_up_convs_modules_2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_up_convs_modules_2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_up_convs_modules_2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_up_convs_modules_2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_up_convs_modules_2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_up_convs_modules_3_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_up_convs_modules_3_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_up_convs_modules_3_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_up_convs_modules_3_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_up_convs_modules_3_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_up_convs_modules_3_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_up_convs_modules_3_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_up_convs_modules_3_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_up_convs_modules_3_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_up_convs_modules_3_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_conv_seg_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_conv_seg_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_inputs_ = L_inputs_
        l_self_modules_backbone_modules_patch_embed_modules_projection_parameters_weight_ = L_self_modules_backbone_modules_patch_embed_modules_projection_parameters_weight_
        l_self_modules_backbone_parameters_cls_token_ = (
            L_self_modules_backbone_parameters_cls_token_
        )
        l_self_modules_backbone_parameters_pos_embed_ = (
            L_self_modules_backbone_parameters_pos_embed_
        )
        l_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_4_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_4_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_4_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_4_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_5_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_5_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_5_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_5_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_6_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_6_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_6_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_6_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_7_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_7_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_7_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_7_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_8_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_8_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_8_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_8_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_9_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_9_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_9_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_9_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_10_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_10_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_10_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_10_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_11_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_11_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_11_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_11_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_12_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_12_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_12_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_12_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_12_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_12_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_12_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_12_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_13_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_13_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_13_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_13_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_13_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_13_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_13_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_13_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_14_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_14_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_14_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_14_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_14_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_14_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_14_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_14_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_15_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_15_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_15_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_15_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_15_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_15_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_15_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_15_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_16_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_16_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_16_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_16_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_16_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_16_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_16_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_16_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_17_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_17_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_17_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_17_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_17_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_17_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_17_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_17_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_18_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_18_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_18_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_18_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_18_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_18_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_18_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_18_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_19_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_19_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_19_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_19_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_19_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_19_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_19_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_19_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_20_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_20_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_20_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_20_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_20_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_20_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_20_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_20_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_21_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_21_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_21_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_21_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_21_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_21_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_21_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_21_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_22_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_22_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_22_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_22_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_22_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_22_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_22_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_22_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_23_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_23_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_23_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_23_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_23_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_23_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_23_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_23_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_neck_modules_norm_modules_0_parameters_weight_ = (
            L_self_modules_neck_modules_norm_modules_0_parameters_weight_
        )
        l_self_modules_neck_modules_norm_modules_0_parameters_bias_ = (
            L_self_modules_neck_modules_norm_modules_0_parameters_bias_
        )
        l_self_modules_neck_modules_norm_modules_1_parameters_weight_ = (
            L_self_modules_neck_modules_norm_modules_1_parameters_weight_
        )
        l_self_modules_neck_modules_norm_modules_1_parameters_bias_ = (
            L_self_modules_neck_modules_norm_modules_1_parameters_bias_
        )
        l_self_modules_neck_modules_norm_modules_2_parameters_weight_ = (
            L_self_modules_neck_modules_norm_modules_2_parameters_weight_
        )
        l_self_modules_neck_modules_norm_modules_2_parameters_bias_ = (
            L_self_modules_neck_modules_norm_modules_2_parameters_bias_
        )
        l_self_modules_neck_modules_norm_modules_3_parameters_weight_ = (
            L_self_modules_neck_modules_norm_modules_3_parameters_weight_
        )
        l_self_modules_neck_modules_norm_modules_3_parameters_bias_ = (
            L_self_modules_neck_modules_norm_modules_3_parameters_bias_
        )
        l_self_modules_neck_modules_mla_modules_channel_proj_modules_0_modules_conv_parameters_weight_ = L_self_modules_neck_modules_mla_modules_channel_proj_modules_0_modules_conv_parameters_weight_
        l_self_modules_neck_modules_mla_modules_channel_proj_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_mla_modules_channel_proj_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_mla_modules_channel_proj_modules_0_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_mla_modules_channel_proj_modules_0_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_mla_modules_channel_proj_modules_0_modules_bn_parameters_weight_ = L_self_modules_neck_modules_mla_modules_channel_proj_modules_0_modules_bn_parameters_weight_
        l_self_modules_neck_modules_mla_modules_channel_proj_modules_0_modules_bn_parameters_bias_ = L_self_modules_neck_modules_mla_modules_channel_proj_modules_0_modules_bn_parameters_bias_
        l_self_modules_neck_modules_mla_modules_channel_proj_modules_1_modules_conv_parameters_weight_ = L_self_modules_neck_modules_mla_modules_channel_proj_modules_1_modules_conv_parameters_weight_
        l_self_modules_neck_modules_mla_modules_channel_proj_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_mla_modules_channel_proj_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_mla_modules_channel_proj_modules_1_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_mla_modules_channel_proj_modules_1_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_mla_modules_channel_proj_modules_1_modules_bn_parameters_weight_ = L_self_modules_neck_modules_mla_modules_channel_proj_modules_1_modules_bn_parameters_weight_
        l_self_modules_neck_modules_mla_modules_channel_proj_modules_1_modules_bn_parameters_bias_ = L_self_modules_neck_modules_mla_modules_channel_proj_modules_1_modules_bn_parameters_bias_
        l_self_modules_neck_modules_mla_modules_channel_proj_modules_2_modules_conv_parameters_weight_ = L_self_modules_neck_modules_mla_modules_channel_proj_modules_2_modules_conv_parameters_weight_
        l_self_modules_neck_modules_mla_modules_channel_proj_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_mla_modules_channel_proj_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_mla_modules_channel_proj_modules_2_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_mla_modules_channel_proj_modules_2_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_mla_modules_channel_proj_modules_2_modules_bn_parameters_weight_ = L_self_modules_neck_modules_mla_modules_channel_proj_modules_2_modules_bn_parameters_weight_
        l_self_modules_neck_modules_mla_modules_channel_proj_modules_2_modules_bn_parameters_bias_ = L_self_modules_neck_modules_mla_modules_channel_proj_modules_2_modules_bn_parameters_bias_
        l_self_modules_neck_modules_mla_modules_channel_proj_modules_3_modules_conv_parameters_weight_ = L_self_modules_neck_modules_mla_modules_channel_proj_modules_3_modules_conv_parameters_weight_
        l_self_modules_neck_modules_mla_modules_channel_proj_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_mla_modules_channel_proj_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_mla_modules_channel_proj_modules_3_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_mla_modules_channel_proj_modules_3_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_mla_modules_channel_proj_modules_3_modules_bn_parameters_weight_ = L_self_modules_neck_modules_mla_modules_channel_proj_modules_3_modules_bn_parameters_weight_
        l_self_modules_neck_modules_mla_modules_channel_proj_modules_3_modules_bn_parameters_bias_ = L_self_modules_neck_modules_mla_modules_channel_proj_modules_3_modules_bn_parameters_bias_
        l_self_modules_neck_modules_mla_modules_feat_extract_modules_0_modules_conv_parameters_weight_ = L_self_modules_neck_modules_mla_modules_feat_extract_modules_0_modules_conv_parameters_weight_
        l_self_modules_neck_modules_mla_modules_feat_extract_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_mla_modules_feat_extract_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_mla_modules_feat_extract_modules_0_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_mla_modules_feat_extract_modules_0_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_mla_modules_feat_extract_modules_0_modules_bn_parameters_weight_ = L_self_modules_neck_modules_mla_modules_feat_extract_modules_0_modules_bn_parameters_weight_
        l_self_modules_neck_modules_mla_modules_feat_extract_modules_0_modules_bn_parameters_bias_ = L_self_modules_neck_modules_mla_modules_feat_extract_modules_0_modules_bn_parameters_bias_
        l_self_modules_neck_modules_mla_modules_feat_extract_modules_1_modules_conv_parameters_weight_ = L_self_modules_neck_modules_mla_modules_feat_extract_modules_1_modules_conv_parameters_weight_
        l_self_modules_neck_modules_mla_modules_feat_extract_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_mla_modules_feat_extract_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_mla_modules_feat_extract_modules_1_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_mla_modules_feat_extract_modules_1_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_mla_modules_feat_extract_modules_1_modules_bn_parameters_weight_ = L_self_modules_neck_modules_mla_modules_feat_extract_modules_1_modules_bn_parameters_weight_
        l_self_modules_neck_modules_mla_modules_feat_extract_modules_1_modules_bn_parameters_bias_ = L_self_modules_neck_modules_mla_modules_feat_extract_modules_1_modules_bn_parameters_bias_
        l_self_modules_neck_modules_mla_modules_feat_extract_modules_2_modules_conv_parameters_weight_ = L_self_modules_neck_modules_mla_modules_feat_extract_modules_2_modules_conv_parameters_weight_
        l_self_modules_neck_modules_mla_modules_feat_extract_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_mla_modules_feat_extract_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_mla_modules_feat_extract_modules_2_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_mla_modules_feat_extract_modules_2_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_mla_modules_feat_extract_modules_2_modules_bn_parameters_weight_ = L_self_modules_neck_modules_mla_modules_feat_extract_modules_2_modules_bn_parameters_weight_
        l_self_modules_neck_modules_mla_modules_feat_extract_modules_2_modules_bn_parameters_bias_ = L_self_modules_neck_modules_mla_modules_feat_extract_modules_2_modules_bn_parameters_bias_
        l_self_modules_neck_modules_mla_modules_feat_extract_modules_3_modules_conv_parameters_weight_ = L_self_modules_neck_modules_mla_modules_feat_extract_modules_3_modules_conv_parameters_weight_
        l_self_modules_neck_modules_mla_modules_feat_extract_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_mla_modules_feat_extract_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_mla_modules_feat_extract_modules_3_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_mla_modules_feat_extract_modules_3_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_mla_modules_feat_extract_modules_3_modules_bn_parameters_weight_ = L_self_modules_neck_modules_mla_modules_feat_extract_modules_3_modules_bn_parameters_weight_
        l_self_modules_neck_modules_mla_modules_feat_extract_modules_3_modules_bn_parameters_bias_ = L_self_modules_neck_modules_mla_modules_feat_extract_modules_3_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_up_convs_modules_0_modules_0_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_up_convs_modules_0_modules_0_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_up_convs_modules_0_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_up_convs_modules_0_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_up_convs_modules_0_modules_0_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_up_convs_modules_0_modules_0_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_up_convs_modules_0_modules_0_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_up_convs_modules_0_modules_0_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_up_convs_modules_0_modules_0_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_up_convs_modules_0_modules_0_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_up_convs_modules_0_modules_1_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_up_convs_modules_0_modules_1_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_up_convs_modules_0_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_up_convs_modules_0_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_up_convs_modules_0_modules_1_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_up_convs_modules_0_modules_1_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_up_convs_modules_0_modules_1_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_up_convs_modules_0_modules_1_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_up_convs_modules_0_modules_1_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_up_convs_modules_0_modules_1_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_up_convs_modules_1_modules_0_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_up_convs_modules_1_modules_0_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_up_convs_modules_1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_up_convs_modules_1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_up_convs_modules_1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_up_convs_modules_1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_up_convs_modules_1_modules_0_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_up_convs_modules_1_modules_0_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_up_convs_modules_1_modules_0_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_up_convs_modules_1_modules_0_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_up_convs_modules_1_modules_1_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_up_convs_modules_1_modules_1_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_up_convs_modules_1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_up_convs_modules_1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_up_convs_modules_1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_up_convs_modules_1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_up_convs_modules_1_modules_1_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_up_convs_modules_1_modules_1_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_up_convs_modules_1_modules_1_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_up_convs_modules_1_modules_1_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_up_convs_modules_2_modules_0_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_up_convs_modules_2_modules_0_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_up_convs_modules_2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_up_convs_modules_2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_up_convs_modules_2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_up_convs_modules_2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_up_convs_modules_2_modules_0_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_up_convs_modules_2_modules_0_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_up_convs_modules_2_modules_0_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_up_convs_modules_2_modules_0_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_up_convs_modules_2_modules_1_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_up_convs_modules_2_modules_1_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_up_convs_modules_2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_up_convs_modules_2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_up_convs_modules_2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_up_convs_modules_2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_up_convs_modules_2_modules_1_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_up_convs_modules_2_modules_1_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_up_convs_modules_2_modules_1_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_up_convs_modules_2_modules_1_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_up_convs_modules_3_modules_0_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_up_convs_modules_3_modules_0_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_up_convs_modules_3_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_up_convs_modules_3_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_up_convs_modules_3_modules_0_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_up_convs_modules_3_modules_0_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_up_convs_modules_3_modules_0_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_up_convs_modules_3_modules_0_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_up_convs_modules_3_modules_0_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_up_convs_modules_3_modules_0_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_up_convs_modules_3_modules_1_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_up_convs_modules_3_modules_1_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_up_convs_modules_3_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_up_convs_modules_3_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_up_convs_modules_3_modules_1_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_up_convs_modules_3_modules_1_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_up_convs_modules_3_modules_1_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_up_convs_modules_3_modules_1_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_up_convs_modules_3_modules_1_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_up_convs_modules_3_modules_1_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_conv_seg_parameters_weight_ = (
            L_self_modules_decode_head_modules_conv_seg_parameters_weight_
        )
        l_self_modules_decode_head_modules_conv_seg_parameters_bias_ = (
            L_self_modules_decode_head_modules_conv_seg_parameters_bias_
        )
        x = torch.conv2d(
            l_inputs_,
            l_self_modules_backbone_modules_patch_embed_modules_projection_parameters_weight_,
            None,
            (16, 16),
            (0, 0),
            (1, 1),
            1,
        )
        l_inputs_ = l_self_modules_backbone_modules_patch_embed_modules_projection_parameters_weight_ = (None)
        flatten = x.flatten(2)
        x = None
        x_1 = flatten.transpose(1, 2)
        flatten = None
        cls_tokens = l_self_modules_backbone_parameters_cls_token_.expand(1, -1, -1)
        l_self_modules_backbone_parameters_cls_token_ = None
        x_2 = torch.cat((cls_tokens, x_1), dim=1)
        cls_tokens = x_1 = None
        add = x_2 + l_self_modules_backbone_parameters_pos_embed_
        x_2 = l_self_modules_backbone_parameters_pos_embed_ = None
        x_3 = torch.nn.functional.dropout(add, 0.0, False, False)
        add = None
        x_4 = x_3[(slice(None, None, None), slice(1, None, None))]
        x_3 = None
        key = torch.nn.functional.layer_norm(
            x_4,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_bias_ = (None)
        query = key.transpose(0, 1)
        key_1 = key.transpose(0, 1)
        value = key.transpose(0, 1)
        key = None
        multi_head_attention_forward = torch.nn.functional.multi_head_attention_forward(
            query,
            key_1,
            value,
            1024,
            16,
            l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query = (
            key_1
        ) = (
            value
        ) = l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output = multi_head_attention_forward[0]
        multi_head_attention_forward = None
        out = attn_output.transpose(0, 1)
        attn_output = None
        dropout_1 = torch.nn.functional.dropout(out, 0.0, False, False)
        out = None
        dropout_2 = torch.nn.functional.dropout(dropout_1, 0.0, False, False)
        dropout_1 = None
        output = x_4 + dropout_2
        x_4 = dropout_2 = None
        layer_norm_1 = torch.nn.functional.layer_norm(
            output,
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
        output_1 = output + input_5
        output = input_5 = None
        key_2 = torch.nn.functional.layer_norm(
            output_1,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_bias_ = (None)
        query_1 = key_2.transpose(0, 1)
        key_3 = key_2.transpose(0, 1)
        value_1 = key_2.transpose(0, 1)
        key_2 = None
        multi_head_attention_forward_1 = torch.nn.functional.multi_head_attention_forward(
            query_1,
            key_3,
            value_1,
            1024,
            16,
            l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query_1 = (
            key_3
        ) = (
            value_1
        ) = l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_1 = multi_head_attention_forward_1[0]
        multi_head_attention_forward_1 = None
        out_1 = attn_output_1.transpose(0, 1)
        attn_output_1 = None
        dropout_5 = torch.nn.functional.dropout(out_1, 0.0, False, False)
        out_1 = None
        dropout_6 = torch.nn.functional.dropout(dropout_5, 0.0, False, False)
        dropout_5 = None
        output_2 = output_1 + dropout_6
        output_1 = dropout_6 = None
        layer_norm_3 = torch.nn.functional.layer_norm(
            output_2,
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
        output_3 = output_2 + input_10
        output_2 = input_10 = None
        key_4 = torch.nn.functional.layer_norm(
            output_3,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_bias_ = (None)
        query_2 = key_4.transpose(0, 1)
        key_5 = key_4.transpose(0, 1)
        value_2 = key_4.transpose(0, 1)
        key_4 = None
        multi_head_attention_forward_2 = torch.nn.functional.multi_head_attention_forward(
            query_2,
            key_5,
            value_2,
            1024,
            16,
            l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query_2 = (
            key_5
        ) = (
            value_2
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_2 = multi_head_attention_forward_2[0]
        multi_head_attention_forward_2 = None
        out_2 = attn_output_2.transpose(0, 1)
        attn_output_2 = None
        dropout_9 = torch.nn.functional.dropout(out_2, 0.0, False, False)
        out_2 = None
        dropout_10 = torch.nn.functional.dropout(dropout_9, 0.0, False, False)
        dropout_9 = None
        output_4 = output_3 + dropout_10
        output_3 = dropout_10 = None
        layer_norm_5 = torch.nn.functional.layer_norm(
            output_4,
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
        output_5 = output_4 + input_15
        output_4 = input_15 = None
        key_6 = torch.nn.functional.layer_norm(
            output_5,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_bias_ = (None)
        query_3 = key_6.transpose(0, 1)
        key_7 = key_6.transpose(0, 1)
        value_3 = key_6.transpose(0, 1)
        key_6 = None
        multi_head_attention_forward_3 = torch.nn.functional.multi_head_attention_forward(
            query_3,
            key_7,
            value_3,
            1024,
            16,
            l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query_3 = (
            key_7
        ) = (
            value_3
        ) = l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_3 = multi_head_attention_forward_3[0]
        multi_head_attention_forward_3 = None
        out_3 = attn_output_3.transpose(0, 1)
        attn_output_3 = None
        dropout_13 = torch.nn.functional.dropout(out_3, 0.0, False, False)
        out_3 = None
        dropout_14 = torch.nn.functional.dropout(dropout_13, 0.0, False, False)
        dropout_13 = None
        output_6 = output_5 + dropout_14
        output_5 = dropout_14 = None
        layer_norm_7 = torch.nn.functional.layer_norm(
            output_6,
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
        output_7 = output_6 + input_20
        output_6 = input_20 = None
        key_8 = torch.nn.functional.layer_norm(
            output_7,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_bias_ = (None)
        query_4 = key_8.transpose(0, 1)
        key_9 = key_8.transpose(0, 1)
        value_4 = key_8.transpose(0, 1)
        key_8 = None
        multi_head_attention_forward_4 = torch.nn.functional.multi_head_attention_forward(
            query_4,
            key_9,
            value_4,
            1024,
            16,
            l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query_4 = (
            key_9
        ) = (
            value_4
        ) = l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_4 = multi_head_attention_forward_4[0]
        multi_head_attention_forward_4 = None
        out_4 = attn_output_4.transpose(0, 1)
        attn_output_4 = None
        dropout_17 = torch.nn.functional.dropout(out_4, 0.0, False, False)
        out_4 = None
        dropout_18 = torch.nn.functional.dropout(dropout_17, 0.0, False, False)
        dropout_17 = None
        output_8 = output_7 + dropout_18
        output_7 = dropout_18 = None
        layer_norm_9 = torch.nn.functional.layer_norm(
            output_8,
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
        output_9 = output_8 + input_25
        output_8 = input_25 = None
        key_10 = torch.nn.functional.layer_norm(
            output_9,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_bias_ = (None)
        query_5 = key_10.transpose(0, 1)
        key_11 = key_10.transpose(0, 1)
        value_5 = key_10.transpose(0, 1)
        key_10 = None
        multi_head_attention_forward_5 = torch.nn.functional.multi_head_attention_forward(
            query_5,
            key_11,
            value_5,
            1024,
            16,
            l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query_5 = (
            key_11
        ) = (
            value_5
        ) = l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_5 = multi_head_attention_forward_5[0]
        multi_head_attention_forward_5 = None
        out_5 = attn_output_5.transpose(0, 1)
        attn_output_5 = None
        dropout_21 = torch.nn.functional.dropout(out_5, 0.0, False, False)
        out_5 = None
        dropout_22 = torch.nn.functional.dropout(dropout_21, 0.0, False, False)
        dropout_21 = None
        output_10 = output_9 + dropout_22
        output_9 = dropout_22 = None
        layer_norm_11 = torch.nn.functional.layer_norm(
            output_10,
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
        output_11 = output_10 + input_30
        output_10 = input_30 = None
        reshape = output_11.reshape(1, 32, 32, 1024)
        permute = reshape.permute(0, 3, 1, 2)
        reshape = None
        out_6 = permute.contiguous()
        permute = None
        key_12 = torch.nn.functional.layer_norm(
            output_11,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_bias_ = (None)
        query_6 = key_12.transpose(0, 1)
        key_13 = key_12.transpose(0, 1)
        value_6 = key_12.transpose(0, 1)
        key_12 = None
        multi_head_attention_forward_6 = torch.nn.functional.multi_head_attention_forward(
            query_6,
            key_13,
            value_6,
            1024,
            16,
            l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query_6 = (
            key_13
        ) = (
            value_6
        ) = l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_6 = multi_head_attention_forward_6[0]
        multi_head_attention_forward_6 = None
        out_7 = attn_output_6.transpose(0, 1)
        attn_output_6 = None
        dropout_25 = torch.nn.functional.dropout(out_7, 0.0, False, False)
        out_7 = None
        dropout_26 = torch.nn.functional.dropout(dropout_25, 0.0, False, False)
        dropout_25 = None
        output_12 = output_11 + dropout_26
        output_11 = dropout_26 = None
        layer_norm_13 = torch.nn.functional.layer_norm(
            output_12,
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
        output_13 = output_12 + input_35
        output_12 = input_35 = None
        key_14 = torch.nn.functional.layer_norm(
            output_13,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_bias_ = (None)
        query_7 = key_14.transpose(0, 1)
        key_15 = key_14.transpose(0, 1)
        value_7 = key_14.transpose(0, 1)
        key_14 = None
        multi_head_attention_forward_7 = torch.nn.functional.multi_head_attention_forward(
            query_7,
            key_15,
            value_7,
            1024,
            16,
            l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query_7 = (
            key_15
        ) = (
            value_7
        ) = l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_7 = multi_head_attention_forward_7[0]
        multi_head_attention_forward_7 = None
        out_8 = attn_output_7.transpose(0, 1)
        attn_output_7 = None
        dropout_29 = torch.nn.functional.dropout(out_8, 0.0, False, False)
        out_8 = None
        dropout_30 = torch.nn.functional.dropout(dropout_29, 0.0, False, False)
        dropout_29 = None
        output_14 = output_13 + dropout_30
        output_13 = dropout_30 = None
        layer_norm_15 = torch.nn.functional.layer_norm(
            output_14,
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
        output_15 = output_14 + input_40
        output_14 = input_40 = None
        key_16 = torch.nn.functional.layer_norm(
            output_15,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_bias_ = (None)
        query_8 = key_16.transpose(0, 1)
        key_17 = key_16.transpose(0, 1)
        value_8 = key_16.transpose(0, 1)
        key_16 = None
        multi_head_attention_forward_8 = torch.nn.functional.multi_head_attention_forward(
            query_8,
            key_17,
            value_8,
            1024,
            16,
            l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query_8 = (
            key_17
        ) = (
            value_8
        ) = l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_8 = multi_head_attention_forward_8[0]
        multi_head_attention_forward_8 = None
        out_9 = attn_output_8.transpose(0, 1)
        attn_output_8 = None
        dropout_33 = torch.nn.functional.dropout(out_9, 0.0, False, False)
        out_9 = None
        dropout_34 = torch.nn.functional.dropout(dropout_33, 0.0, False, False)
        dropout_33 = None
        output_16 = output_15 + dropout_34
        output_15 = dropout_34 = None
        layer_norm_17 = torch.nn.functional.layer_norm(
            output_16,
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
        output_17 = output_16 + input_45
        output_16 = input_45 = None
        key_18 = torch.nn.functional.layer_norm(
            output_17,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_bias_ = (None)
        query_9 = key_18.transpose(0, 1)
        key_19 = key_18.transpose(0, 1)
        value_9 = key_18.transpose(0, 1)
        key_18 = None
        multi_head_attention_forward_9 = torch.nn.functional.multi_head_attention_forward(
            query_9,
            key_19,
            value_9,
            1024,
            16,
            l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query_9 = (
            key_19
        ) = (
            value_9
        ) = l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_9 = multi_head_attention_forward_9[0]
        multi_head_attention_forward_9 = None
        out_10 = attn_output_9.transpose(0, 1)
        attn_output_9 = None
        dropout_37 = torch.nn.functional.dropout(out_10, 0.0, False, False)
        out_10 = None
        dropout_38 = torch.nn.functional.dropout(dropout_37, 0.0, False, False)
        dropout_37 = None
        output_18 = output_17 + dropout_38
        output_17 = dropout_38 = None
        layer_norm_19 = torch.nn.functional.layer_norm(
            output_18,
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
        output_19 = output_18 + input_50
        output_18 = input_50 = None
        key_20 = torch.nn.functional.layer_norm(
            output_19,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_bias_ = (None)
        query_10 = key_20.transpose(0, 1)
        key_21 = key_20.transpose(0, 1)
        value_10 = key_20.transpose(0, 1)
        key_20 = None
        multi_head_attention_forward_10 = torch.nn.functional.multi_head_attention_forward(
            query_10,
            key_21,
            value_10,
            1024,
            16,
            l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query_10 = (
            key_21
        ) = (
            value_10
        ) = l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_10 = multi_head_attention_forward_10[0]
        multi_head_attention_forward_10 = None
        out_11 = attn_output_10.transpose(0, 1)
        attn_output_10 = None
        dropout_41 = torch.nn.functional.dropout(out_11, 0.0, False, False)
        out_11 = None
        dropout_42 = torch.nn.functional.dropout(dropout_41, 0.0, False, False)
        dropout_41 = None
        output_20 = output_19 + dropout_42
        output_19 = dropout_42 = None
        layer_norm_21 = torch.nn.functional.layer_norm(
            output_20,
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
        output_21 = output_20 + input_55
        output_20 = input_55 = None
        key_22 = torch.nn.functional.layer_norm(
            output_21,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_bias_ = (None)
        query_11 = key_22.transpose(0, 1)
        key_23 = key_22.transpose(0, 1)
        value_11 = key_22.transpose(0, 1)
        key_22 = None
        multi_head_attention_forward_11 = torch.nn.functional.multi_head_attention_forward(
            query_11,
            key_23,
            value_11,
            1024,
            16,
            l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query_11 = (
            key_23
        ) = (
            value_11
        ) = l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_11 = multi_head_attention_forward_11[0]
        multi_head_attention_forward_11 = None
        out_12 = attn_output_11.transpose(0, 1)
        attn_output_11 = None
        dropout_45 = torch.nn.functional.dropout(out_12, 0.0, False, False)
        out_12 = None
        dropout_46 = torch.nn.functional.dropout(dropout_45, 0.0, False, False)
        dropout_45 = None
        output_22 = output_21 + dropout_46
        output_21 = dropout_46 = None
        layer_norm_23 = torch.nn.functional.layer_norm(
            output_22,
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
        output_23 = output_22 + input_60
        output_22 = input_60 = None
        reshape_1 = output_23.reshape(1, 32, 32, 1024)
        permute_1 = reshape_1.permute(0, 3, 1, 2)
        reshape_1 = None
        out_13 = permute_1.contiguous()
        permute_1 = None
        key_24 = torch.nn.functional.layer_norm(
            output_23,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_12_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_12_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_12_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_12_modules_ln1_parameters_bias_ = (None)
        query_12 = key_24.transpose(0, 1)
        key_25 = key_24.transpose(0, 1)
        value_12 = key_24.transpose(0, 1)
        key_24 = None
        multi_head_attention_forward_12 = torch.nn.functional.multi_head_attention_forward(
            query_12,
            key_25,
            value_12,
            1024,
            16,
            l_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query_12 = (
            key_25
        ) = (
            value_12
        ) = l_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_12 = multi_head_attention_forward_12[0]
        multi_head_attention_forward_12 = None
        out_14 = attn_output_12.transpose(0, 1)
        attn_output_12 = None
        dropout_49 = torch.nn.functional.dropout(out_14, 0.0, False, False)
        out_14 = None
        dropout_50 = torch.nn.functional.dropout(dropout_49, 0.0, False, False)
        dropout_49 = None
        output_24 = output_23 + dropout_50
        output_23 = dropout_50 = None
        layer_norm_25 = torch.nn.functional.layer_norm(
            output_24,
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
        output_25 = output_24 + input_65
        output_24 = input_65 = None
        key_26 = torch.nn.functional.layer_norm(
            output_25,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_13_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_13_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_13_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_13_modules_ln1_parameters_bias_ = (None)
        query_13 = key_26.transpose(0, 1)
        key_27 = key_26.transpose(0, 1)
        value_13 = key_26.transpose(0, 1)
        key_26 = None
        multi_head_attention_forward_13 = torch.nn.functional.multi_head_attention_forward(
            query_13,
            key_27,
            value_13,
            1024,
            16,
            l_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query_13 = (
            key_27
        ) = (
            value_13
        ) = l_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_13 = multi_head_attention_forward_13[0]
        multi_head_attention_forward_13 = None
        out_15 = attn_output_13.transpose(0, 1)
        attn_output_13 = None
        dropout_53 = torch.nn.functional.dropout(out_15, 0.0, False, False)
        out_15 = None
        dropout_54 = torch.nn.functional.dropout(dropout_53, 0.0, False, False)
        dropout_53 = None
        output_26 = output_25 + dropout_54
        output_25 = dropout_54 = None
        layer_norm_27 = torch.nn.functional.layer_norm(
            output_26,
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
        output_27 = output_26 + input_70
        output_26 = input_70 = None
        key_28 = torch.nn.functional.layer_norm(
            output_27,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_14_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_14_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_14_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_14_modules_ln1_parameters_bias_ = (None)
        query_14 = key_28.transpose(0, 1)
        key_29 = key_28.transpose(0, 1)
        value_14 = key_28.transpose(0, 1)
        key_28 = None
        multi_head_attention_forward_14 = torch.nn.functional.multi_head_attention_forward(
            query_14,
            key_29,
            value_14,
            1024,
            16,
            l_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query_14 = (
            key_29
        ) = (
            value_14
        ) = l_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_14 = multi_head_attention_forward_14[0]
        multi_head_attention_forward_14 = None
        out_16 = attn_output_14.transpose(0, 1)
        attn_output_14 = None
        dropout_57 = torch.nn.functional.dropout(out_16, 0.0, False, False)
        out_16 = None
        dropout_58 = torch.nn.functional.dropout(dropout_57, 0.0, False, False)
        dropout_57 = None
        output_28 = output_27 + dropout_58
        output_27 = dropout_58 = None
        layer_norm_29 = torch.nn.functional.layer_norm(
            output_28,
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
        output_29 = output_28 + input_75
        output_28 = input_75 = None
        key_30 = torch.nn.functional.layer_norm(
            output_29,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_15_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_15_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_15_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_15_modules_ln1_parameters_bias_ = (None)
        query_15 = key_30.transpose(0, 1)
        key_31 = key_30.transpose(0, 1)
        value_15 = key_30.transpose(0, 1)
        key_30 = None
        multi_head_attention_forward_15 = torch.nn.functional.multi_head_attention_forward(
            query_15,
            key_31,
            value_15,
            1024,
            16,
            l_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query_15 = (
            key_31
        ) = (
            value_15
        ) = l_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_15 = multi_head_attention_forward_15[0]
        multi_head_attention_forward_15 = None
        out_17 = attn_output_15.transpose(0, 1)
        attn_output_15 = None
        dropout_61 = torch.nn.functional.dropout(out_17, 0.0, False, False)
        out_17 = None
        dropout_62 = torch.nn.functional.dropout(dropout_61, 0.0, False, False)
        dropout_61 = None
        output_30 = output_29 + dropout_62
        output_29 = dropout_62 = None
        layer_norm_31 = torch.nn.functional.layer_norm(
            output_30,
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
        output_31 = output_30 + input_80
        output_30 = input_80 = None
        key_32 = torch.nn.functional.layer_norm(
            output_31,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_16_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_16_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_16_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_16_modules_ln1_parameters_bias_ = (None)
        query_16 = key_32.transpose(0, 1)
        key_33 = key_32.transpose(0, 1)
        value_16 = key_32.transpose(0, 1)
        key_32 = None
        multi_head_attention_forward_16 = torch.nn.functional.multi_head_attention_forward(
            query_16,
            key_33,
            value_16,
            1024,
            16,
            l_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query_16 = (
            key_33
        ) = (
            value_16
        ) = l_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_16 = multi_head_attention_forward_16[0]
        multi_head_attention_forward_16 = None
        out_18 = attn_output_16.transpose(0, 1)
        attn_output_16 = None
        dropout_65 = torch.nn.functional.dropout(out_18, 0.0, False, False)
        out_18 = None
        dropout_66 = torch.nn.functional.dropout(dropout_65, 0.0, False, False)
        dropout_65 = None
        output_32 = output_31 + dropout_66
        output_31 = dropout_66 = None
        layer_norm_33 = torch.nn.functional.layer_norm(
            output_32,
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
        output_33 = output_32 + input_85
        output_32 = input_85 = None
        key_34 = torch.nn.functional.layer_norm(
            output_33,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_17_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_17_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_17_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_17_modules_ln1_parameters_bias_ = (None)
        query_17 = key_34.transpose(0, 1)
        key_35 = key_34.transpose(0, 1)
        value_17 = key_34.transpose(0, 1)
        key_34 = None
        multi_head_attention_forward_17 = torch.nn.functional.multi_head_attention_forward(
            query_17,
            key_35,
            value_17,
            1024,
            16,
            l_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query_17 = (
            key_35
        ) = (
            value_17
        ) = l_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_17 = multi_head_attention_forward_17[0]
        multi_head_attention_forward_17 = None
        out_19 = attn_output_17.transpose(0, 1)
        attn_output_17 = None
        dropout_69 = torch.nn.functional.dropout(out_19, 0.0, False, False)
        out_19 = None
        dropout_70 = torch.nn.functional.dropout(dropout_69, 0.0, False, False)
        dropout_69 = None
        output_34 = output_33 + dropout_70
        output_33 = dropout_70 = None
        layer_norm_35 = torch.nn.functional.layer_norm(
            output_34,
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
        output_35 = output_34 + input_90
        output_34 = input_90 = None
        reshape_2 = output_35.reshape(1, 32, 32, 1024)
        permute_2 = reshape_2.permute(0, 3, 1, 2)
        reshape_2 = None
        out_20 = permute_2.contiguous()
        permute_2 = None
        key_36 = torch.nn.functional.layer_norm(
            output_35,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_18_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_18_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_18_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_18_modules_ln1_parameters_bias_ = (None)
        query_18 = key_36.transpose(0, 1)
        key_37 = key_36.transpose(0, 1)
        value_18 = key_36.transpose(0, 1)
        key_36 = None
        multi_head_attention_forward_18 = torch.nn.functional.multi_head_attention_forward(
            query_18,
            key_37,
            value_18,
            1024,
            16,
            l_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query_18 = (
            key_37
        ) = (
            value_18
        ) = l_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_18 = multi_head_attention_forward_18[0]
        multi_head_attention_forward_18 = None
        out_21 = attn_output_18.transpose(0, 1)
        attn_output_18 = None
        dropout_73 = torch.nn.functional.dropout(out_21, 0.0, False, False)
        out_21 = None
        dropout_74 = torch.nn.functional.dropout(dropout_73, 0.0, False, False)
        dropout_73 = None
        output_36 = output_35 + dropout_74
        output_35 = dropout_74 = None
        layer_norm_37 = torch.nn.functional.layer_norm(
            output_36,
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
        output_37 = output_36 + input_95
        output_36 = input_95 = None
        key_38 = torch.nn.functional.layer_norm(
            output_37,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_19_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_19_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_19_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_19_modules_ln1_parameters_bias_ = (None)
        query_19 = key_38.transpose(0, 1)
        key_39 = key_38.transpose(0, 1)
        value_19 = key_38.transpose(0, 1)
        key_38 = None
        multi_head_attention_forward_19 = torch.nn.functional.multi_head_attention_forward(
            query_19,
            key_39,
            value_19,
            1024,
            16,
            l_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query_19 = (
            key_39
        ) = (
            value_19
        ) = l_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_19 = multi_head_attention_forward_19[0]
        multi_head_attention_forward_19 = None
        out_22 = attn_output_19.transpose(0, 1)
        attn_output_19 = None
        dropout_77 = torch.nn.functional.dropout(out_22, 0.0, False, False)
        out_22 = None
        dropout_78 = torch.nn.functional.dropout(dropout_77, 0.0, False, False)
        dropout_77 = None
        output_38 = output_37 + dropout_78
        output_37 = dropout_78 = None
        layer_norm_39 = torch.nn.functional.layer_norm(
            output_38,
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
        output_39 = output_38 + input_100
        output_38 = input_100 = None
        key_40 = torch.nn.functional.layer_norm(
            output_39,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_20_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_20_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_20_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_20_modules_ln1_parameters_bias_ = (None)
        query_20 = key_40.transpose(0, 1)
        key_41 = key_40.transpose(0, 1)
        value_20 = key_40.transpose(0, 1)
        key_40 = None
        multi_head_attention_forward_20 = torch.nn.functional.multi_head_attention_forward(
            query_20,
            key_41,
            value_20,
            1024,
            16,
            l_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query_20 = (
            key_41
        ) = (
            value_20
        ) = l_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_20 = multi_head_attention_forward_20[0]
        multi_head_attention_forward_20 = None
        out_23 = attn_output_20.transpose(0, 1)
        attn_output_20 = None
        dropout_81 = torch.nn.functional.dropout(out_23, 0.0, False, False)
        out_23 = None
        dropout_82 = torch.nn.functional.dropout(dropout_81, 0.0, False, False)
        dropout_81 = None
        output_40 = output_39 + dropout_82
        output_39 = dropout_82 = None
        layer_norm_41 = torch.nn.functional.layer_norm(
            output_40,
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
        output_41 = output_40 + input_105
        output_40 = input_105 = None
        key_42 = torch.nn.functional.layer_norm(
            output_41,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_21_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_21_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_21_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_21_modules_ln1_parameters_bias_ = (None)
        query_21 = key_42.transpose(0, 1)
        key_43 = key_42.transpose(0, 1)
        value_21 = key_42.transpose(0, 1)
        key_42 = None
        multi_head_attention_forward_21 = torch.nn.functional.multi_head_attention_forward(
            query_21,
            key_43,
            value_21,
            1024,
            16,
            l_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query_21 = (
            key_43
        ) = (
            value_21
        ) = l_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_21 = multi_head_attention_forward_21[0]
        multi_head_attention_forward_21 = None
        out_24 = attn_output_21.transpose(0, 1)
        attn_output_21 = None
        dropout_85 = torch.nn.functional.dropout(out_24, 0.0, False, False)
        out_24 = None
        dropout_86 = torch.nn.functional.dropout(dropout_85, 0.0, False, False)
        dropout_85 = None
        output_42 = output_41 + dropout_86
        output_41 = dropout_86 = None
        layer_norm_43 = torch.nn.functional.layer_norm(
            output_42,
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
        output_43 = output_42 + input_110
        output_42 = input_110 = None
        key_44 = torch.nn.functional.layer_norm(
            output_43,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_22_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_22_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_22_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_22_modules_ln1_parameters_bias_ = (None)
        query_22 = key_44.transpose(0, 1)
        key_45 = key_44.transpose(0, 1)
        value_22 = key_44.transpose(0, 1)
        key_44 = None
        multi_head_attention_forward_22 = torch.nn.functional.multi_head_attention_forward(
            query_22,
            key_45,
            value_22,
            1024,
            16,
            l_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query_22 = (
            key_45
        ) = (
            value_22
        ) = l_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_22 = multi_head_attention_forward_22[0]
        multi_head_attention_forward_22 = None
        out_25 = attn_output_22.transpose(0, 1)
        attn_output_22 = None
        dropout_89 = torch.nn.functional.dropout(out_25, 0.0, False, False)
        out_25 = None
        dropout_90 = torch.nn.functional.dropout(dropout_89, 0.0, False, False)
        dropout_89 = None
        output_44 = output_43 + dropout_90
        output_43 = dropout_90 = None
        layer_norm_45 = torch.nn.functional.layer_norm(
            output_44,
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
        output_45 = output_44 + input_115
        output_44 = input_115 = None
        key_46 = torch.nn.functional.layer_norm(
            output_45,
            (1024,),
            l_self_modules_backbone_modules_layers_modules_23_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_23_modules_ln1_parameters_bias_,
            1e-06,
        )
        l_self_modules_backbone_modules_layers_modules_23_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_23_modules_ln1_parameters_bias_ = (None)
        query_23 = key_46.transpose(0, 1)
        key_47 = key_46.transpose(0, 1)
        value_23 = key_46.transpose(0, 1)
        key_46 = None
        multi_head_attention_forward_23 = torch.nn.functional.multi_head_attention_forward(
            query_23,
            key_47,
            value_23,
            1024,
            16,
            l_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query_23 = (
            key_47
        ) = (
            value_23
        ) = l_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_23 = multi_head_attention_forward_23[0]
        multi_head_attention_forward_23 = None
        out_26 = attn_output_23.transpose(0, 1)
        attn_output_23 = None
        dropout_93 = torch.nn.functional.dropout(out_26, 0.0, False, False)
        out_26 = None
        dropout_94 = torch.nn.functional.dropout(dropout_93, 0.0, False, False)
        dropout_93 = None
        output_46 = output_45 + dropout_94
        output_45 = dropout_94 = None
        layer_norm_47 = torch.nn.functional.layer_norm(
            output_46,
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
        output_47 = output_46 + input_120
        output_46 = input_120 = None
        reshape_3 = output_47.reshape(1, 32, 32, 1024)
        output_47 = None
        permute_3 = reshape_3.permute(0, 3, 1, 2)
        reshape_3 = None
        out_27 = permute_3.contiguous()
        permute_3 = None
        reshape_4 = out_6.reshape(1, 1024, 1024)
        out_6 = None
        transpose_97 = reshape_4.transpose(2, 1)
        reshape_4 = None
        x_5 = transpose_97.contiguous()
        transpose_97 = None
        x_6 = torch.nn.functional.layer_norm(
            x_5,
            (1024,),
            l_self_modules_neck_modules_norm_modules_0_parameters_weight_,
            l_self_modules_neck_modules_norm_modules_0_parameters_bias_,
            1e-06,
        )
        x_5 = (
            l_self_modules_neck_modules_norm_modules_0_parameters_weight_
        ) = l_self_modules_neck_modules_norm_modules_0_parameters_bias_ = None
        transpose_98 = x_6.transpose(1, 2)
        x_6 = None
        reshape_5 = transpose_98.reshape(1, 1024, 32, 32)
        transpose_98 = None
        x_7 = reshape_5.contiguous()
        reshape_5 = None
        reshape_6 = out_13.reshape(1, 1024, 1024)
        out_13 = None
        transpose_99 = reshape_6.transpose(2, 1)
        reshape_6 = None
        x_8 = transpose_99.contiguous()
        transpose_99 = None
        x_9 = torch.nn.functional.layer_norm(
            x_8,
            (1024,),
            l_self_modules_neck_modules_norm_modules_1_parameters_weight_,
            l_self_modules_neck_modules_norm_modules_1_parameters_bias_,
            1e-06,
        )
        x_8 = (
            l_self_modules_neck_modules_norm_modules_1_parameters_weight_
        ) = l_self_modules_neck_modules_norm_modules_1_parameters_bias_ = None
        transpose_100 = x_9.transpose(1, 2)
        x_9 = None
        reshape_7 = transpose_100.reshape(1, 1024, 32, 32)
        transpose_100 = None
        x_10 = reshape_7.contiguous()
        reshape_7 = None
        reshape_8 = out_20.reshape(1, 1024, 1024)
        out_20 = None
        transpose_101 = reshape_8.transpose(2, 1)
        reshape_8 = None
        x_11 = transpose_101.contiguous()
        transpose_101 = None
        x_12 = torch.nn.functional.layer_norm(
            x_11,
            (1024,),
            l_self_modules_neck_modules_norm_modules_2_parameters_weight_,
            l_self_modules_neck_modules_norm_modules_2_parameters_bias_,
            1e-06,
        )
        x_11 = (
            l_self_modules_neck_modules_norm_modules_2_parameters_weight_
        ) = l_self_modules_neck_modules_norm_modules_2_parameters_bias_ = None
        transpose_102 = x_12.transpose(1, 2)
        x_12 = None
        reshape_9 = transpose_102.reshape(1, 1024, 32, 32)
        transpose_102 = None
        x_13 = reshape_9.contiguous()
        reshape_9 = None
        reshape_10 = out_27.reshape(1, 1024, 1024)
        out_27 = None
        transpose_103 = reshape_10.transpose(2, 1)
        reshape_10 = None
        x_14 = transpose_103.contiguous()
        transpose_103 = None
        x_15 = torch.nn.functional.layer_norm(
            x_14,
            (1024,),
            l_self_modules_neck_modules_norm_modules_3_parameters_weight_,
            l_self_modules_neck_modules_norm_modules_3_parameters_bias_,
            1e-06,
        )
        x_14 = (
            l_self_modules_neck_modules_norm_modules_3_parameters_weight_
        ) = l_self_modules_neck_modules_norm_modules_3_parameters_bias_ = None
        transpose_104 = x_15.transpose(1, 2)
        x_15 = None
        reshape_11 = transpose_104.reshape(1, 1024, 32, 32)
        transpose_104 = None
        x_16 = reshape_11.contiguous()
        reshape_11 = None
        x_17 = torch.conv2d(
            x_7,
            l_self_modules_neck_modules_mla_modules_channel_proj_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_7 = l_self_modules_neck_modules_mla_modules_channel_proj_modules_0_modules_conv_parameters_weight_ = (None)
        x_18 = torch.nn.functional.batch_norm(
            x_17,
            l_self_modules_neck_modules_mla_modules_channel_proj_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_mla_modules_channel_proj_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_mla_modules_channel_proj_modules_0_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_mla_modules_channel_proj_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_17 = l_self_modules_neck_modules_mla_modules_channel_proj_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_mla_modules_channel_proj_modules_0_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_mla_modules_channel_proj_modules_0_modules_bn_parameters_weight_ = l_self_modules_neck_modules_mla_modules_channel_proj_modules_0_modules_bn_parameters_bias_ = (None)
        x_19 = torch.nn.functional.relu(x_18, inplace=True)
        x_18 = None
        x_20 = torch.conv2d(
            x_10,
            l_self_modules_neck_modules_mla_modules_channel_proj_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_10 = l_self_modules_neck_modules_mla_modules_channel_proj_modules_1_modules_conv_parameters_weight_ = (None)
        x_21 = torch.nn.functional.batch_norm(
            x_20,
            l_self_modules_neck_modules_mla_modules_channel_proj_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_mla_modules_channel_proj_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_mla_modules_channel_proj_modules_1_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_mla_modules_channel_proj_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_20 = l_self_modules_neck_modules_mla_modules_channel_proj_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_mla_modules_channel_proj_modules_1_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_mla_modules_channel_proj_modules_1_modules_bn_parameters_weight_ = l_self_modules_neck_modules_mla_modules_channel_proj_modules_1_modules_bn_parameters_bias_ = (None)
        x_22 = torch.nn.functional.relu(x_21, inplace=True)
        x_21 = None
        x_23 = torch.conv2d(
            x_13,
            l_self_modules_neck_modules_mla_modules_channel_proj_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_13 = l_self_modules_neck_modules_mla_modules_channel_proj_modules_2_modules_conv_parameters_weight_ = (None)
        x_24 = torch.nn.functional.batch_norm(
            x_23,
            l_self_modules_neck_modules_mla_modules_channel_proj_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_mla_modules_channel_proj_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_mla_modules_channel_proj_modules_2_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_mla_modules_channel_proj_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_23 = l_self_modules_neck_modules_mla_modules_channel_proj_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_mla_modules_channel_proj_modules_2_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_mla_modules_channel_proj_modules_2_modules_bn_parameters_weight_ = l_self_modules_neck_modules_mla_modules_channel_proj_modules_2_modules_bn_parameters_bias_ = (None)
        x_25 = torch.nn.functional.relu(x_24, inplace=True)
        x_24 = None
        x_26 = torch.conv2d(
            x_16,
            l_self_modules_neck_modules_mla_modules_channel_proj_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_16 = l_self_modules_neck_modules_mla_modules_channel_proj_modules_3_modules_conv_parameters_weight_ = (None)
        x_27 = torch.nn.functional.batch_norm(
            x_26,
            l_self_modules_neck_modules_mla_modules_channel_proj_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_mla_modules_channel_proj_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_mla_modules_channel_proj_modules_3_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_mla_modules_channel_proj_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_26 = l_self_modules_neck_modules_mla_modules_channel_proj_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_mla_modules_channel_proj_modules_3_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_mla_modules_channel_proj_modules_3_modules_bn_parameters_weight_ = l_self_modules_neck_modules_mla_modules_channel_proj_modules_3_modules_bn_parameters_bias_ = (None)
        x_28 = torch.nn.functional.relu(x_27, inplace=True)
        x_27 = None
        mid = x_28 + x_25
        x_25 = None
        mid_1 = mid + x_22
        x_22 = None
        mid_2 = mid_1 + x_19
        x_19 = None
        x_29 = torch.conv2d(
            x_28,
            l_self_modules_neck_modules_mla_modules_feat_extract_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_28 = l_self_modules_neck_modules_mla_modules_feat_extract_modules_0_modules_conv_parameters_weight_ = (None)
        x_30 = torch.nn.functional.batch_norm(
            x_29,
            l_self_modules_neck_modules_mla_modules_feat_extract_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_mla_modules_feat_extract_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_mla_modules_feat_extract_modules_0_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_mla_modules_feat_extract_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_29 = l_self_modules_neck_modules_mla_modules_feat_extract_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_mla_modules_feat_extract_modules_0_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_mla_modules_feat_extract_modules_0_modules_bn_parameters_weight_ = l_self_modules_neck_modules_mla_modules_feat_extract_modules_0_modules_bn_parameters_bias_ = (None)
        x_31 = torch.nn.functional.relu(x_30, inplace=True)
        x_30 = None
        x_32 = torch.conv2d(
            mid,
            l_self_modules_neck_modules_mla_modules_feat_extract_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        mid = l_self_modules_neck_modules_mla_modules_feat_extract_modules_1_modules_conv_parameters_weight_ = (None)
        x_33 = torch.nn.functional.batch_norm(
            x_32,
            l_self_modules_neck_modules_mla_modules_feat_extract_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_mla_modules_feat_extract_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_mla_modules_feat_extract_modules_1_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_mla_modules_feat_extract_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_32 = l_self_modules_neck_modules_mla_modules_feat_extract_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_mla_modules_feat_extract_modules_1_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_mla_modules_feat_extract_modules_1_modules_bn_parameters_weight_ = l_self_modules_neck_modules_mla_modules_feat_extract_modules_1_modules_bn_parameters_bias_ = (None)
        x_34 = torch.nn.functional.relu(x_33, inplace=True)
        x_33 = None
        x_35 = torch.conv2d(
            mid_1,
            l_self_modules_neck_modules_mla_modules_feat_extract_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        mid_1 = l_self_modules_neck_modules_mla_modules_feat_extract_modules_2_modules_conv_parameters_weight_ = (None)
        x_36 = torch.nn.functional.batch_norm(
            x_35,
            l_self_modules_neck_modules_mla_modules_feat_extract_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_mla_modules_feat_extract_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_mla_modules_feat_extract_modules_2_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_mla_modules_feat_extract_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_35 = l_self_modules_neck_modules_mla_modules_feat_extract_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_mla_modules_feat_extract_modules_2_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_mla_modules_feat_extract_modules_2_modules_bn_parameters_weight_ = l_self_modules_neck_modules_mla_modules_feat_extract_modules_2_modules_bn_parameters_bias_ = (None)
        x_37 = torch.nn.functional.relu(x_36, inplace=True)
        x_36 = None
        x_38 = torch.conv2d(
            mid_2,
            l_self_modules_neck_modules_mla_modules_feat_extract_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        mid_2 = l_self_modules_neck_modules_mla_modules_feat_extract_modules_3_modules_conv_parameters_weight_ = (None)
        x_39 = torch.nn.functional.batch_norm(
            x_38,
            l_self_modules_neck_modules_mla_modules_feat_extract_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_mla_modules_feat_extract_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_mla_modules_feat_extract_modules_3_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_mla_modules_feat_extract_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_38 = l_self_modules_neck_modules_mla_modules_feat_extract_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_mla_modules_feat_extract_modules_3_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_mla_modules_feat_extract_modules_3_modules_bn_parameters_weight_ = l_self_modules_neck_modules_mla_modules_feat_extract_modules_3_modules_bn_parameters_bias_ = (None)
        x_40 = torch.nn.functional.relu(x_39, inplace=True)
        x_39 = None
        x_41 = torch.conv2d(
            x_31,
            l_self_modules_decode_head_modules_up_convs_modules_0_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_31 = l_self_modules_decode_head_modules_up_convs_modules_0_modules_0_modules_conv_parameters_weight_ = (None)
        x_42 = torch.nn.functional.batch_norm(
            x_41,
            l_self_modules_decode_head_modules_up_convs_modules_0_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_up_convs_modules_0_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_up_convs_modules_0_modules_0_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_up_convs_modules_0_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_41 = l_self_modules_decode_head_modules_up_convs_modules_0_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_up_convs_modules_0_modules_0_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_up_convs_modules_0_modules_0_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_up_convs_modules_0_modules_0_modules_bn_parameters_bias_ = (None)
        x_43 = torch.nn.functional.relu(x_42, inplace=True)
        x_42 = None
        x_44 = torch.conv2d(
            x_43,
            l_self_modules_decode_head_modules_up_convs_modules_0_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_43 = l_self_modules_decode_head_modules_up_convs_modules_0_modules_1_modules_conv_parameters_weight_ = (None)
        x_45 = torch.nn.functional.batch_norm(
            x_44,
            l_self_modules_decode_head_modules_up_convs_modules_0_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_up_convs_modules_0_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_up_convs_modules_0_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_up_convs_modules_0_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_44 = l_self_modules_decode_head_modules_up_convs_modules_0_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_up_convs_modules_0_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_up_convs_modules_0_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_up_convs_modules_0_modules_1_modules_bn_parameters_bias_ = (None)
        x_46 = torch.nn.functional.relu(x_45, inplace=True)
        x_45 = None
        input_121 = torch.nn.functional.interpolate(
            x_46, [128, 128], None, "bilinear", False
        )
        x_46 = None
        x_47 = torch.conv2d(
            x_34,
            l_self_modules_decode_head_modules_up_convs_modules_1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_34 = l_self_modules_decode_head_modules_up_convs_modules_1_modules_0_modules_conv_parameters_weight_ = (None)
        x_48 = torch.nn.functional.batch_norm(
            x_47,
            l_self_modules_decode_head_modules_up_convs_modules_1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_up_convs_modules_1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_up_convs_modules_1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_up_convs_modules_1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_47 = l_self_modules_decode_head_modules_up_convs_modules_1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_up_convs_modules_1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_up_convs_modules_1_modules_0_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_up_convs_modules_1_modules_0_modules_bn_parameters_bias_ = (None)
        x_49 = torch.nn.functional.relu(x_48, inplace=True)
        x_48 = None
        x_50 = torch.conv2d(
            x_49,
            l_self_modules_decode_head_modules_up_convs_modules_1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_49 = l_self_modules_decode_head_modules_up_convs_modules_1_modules_1_modules_conv_parameters_weight_ = (None)
        x_51 = torch.nn.functional.batch_norm(
            x_50,
            l_self_modules_decode_head_modules_up_convs_modules_1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_up_convs_modules_1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_up_convs_modules_1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_up_convs_modules_1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_50 = l_self_modules_decode_head_modules_up_convs_modules_1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_up_convs_modules_1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_up_convs_modules_1_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_up_convs_modules_1_modules_1_modules_bn_parameters_bias_ = (None)
        x_52 = torch.nn.functional.relu(x_51, inplace=True)
        x_51 = None
        input_122 = torch.nn.functional.interpolate(
            x_52, [128, 128], None, "bilinear", False
        )
        x_52 = None
        x_53 = torch.conv2d(
            x_37,
            l_self_modules_decode_head_modules_up_convs_modules_2_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_37 = l_self_modules_decode_head_modules_up_convs_modules_2_modules_0_modules_conv_parameters_weight_ = (None)
        x_54 = torch.nn.functional.batch_norm(
            x_53,
            l_self_modules_decode_head_modules_up_convs_modules_2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_up_convs_modules_2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_up_convs_modules_2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_up_convs_modules_2_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_53 = l_self_modules_decode_head_modules_up_convs_modules_2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_up_convs_modules_2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_up_convs_modules_2_modules_0_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_up_convs_modules_2_modules_0_modules_bn_parameters_bias_ = (None)
        x_55 = torch.nn.functional.relu(x_54, inplace=True)
        x_54 = None
        x_56 = torch.conv2d(
            x_55,
            l_self_modules_decode_head_modules_up_convs_modules_2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_55 = l_self_modules_decode_head_modules_up_convs_modules_2_modules_1_modules_conv_parameters_weight_ = (None)
        x_57 = torch.nn.functional.batch_norm(
            x_56,
            l_self_modules_decode_head_modules_up_convs_modules_2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_up_convs_modules_2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_up_convs_modules_2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_up_convs_modules_2_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_56 = l_self_modules_decode_head_modules_up_convs_modules_2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_up_convs_modules_2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_up_convs_modules_2_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_up_convs_modules_2_modules_1_modules_bn_parameters_bias_ = (None)
        x_58 = torch.nn.functional.relu(x_57, inplace=True)
        x_57 = None
        input_123 = torch.nn.functional.interpolate(
            x_58, [128, 128], None, "bilinear", False
        )
        x_58 = None
        x_59 = torch.conv2d(
            x_40,
            l_self_modules_decode_head_modules_up_convs_modules_3_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_40 = l_self_modules_decode_head_modules_up_convs_modules_3_modules_0_modules_conv_parameters_weight_ = (None)
        x_60 = torch.nn.functional.batch_norm(
            x_59,
            l_self_modules_decode_head_modules_up_convs_modules_3_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_up_convs_modules_3_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_up_convs_modules_3_modules_0_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_up_convs_modules_3_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_59 = l_self_modules_decode_head_modules_up_convs_modules_3_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_up_convs_modules_3_modules_0_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_up_convs_modules_3_modules_0_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_up_convs_modules_3_modules_0_modules_bn_parameters_bias_ = (None)
        x_61 = torch.nn.functional.relu(x_60, inplace=True)
        x_60 = None
        x_62 = torch.conv2d(
            x_61,
            l_self_modules_decode_head_modules_up_convs_modules_3_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_61 = l_self_modules_decode_head_modules_up_convs_modules_3_modules_1_modules_conv_parameters_weight_ = (None)
        x_63 = torch.nn.functional.batch_norm(
            x_62,
            l_self_modules_decode_head_modules_up_convs_modules_3_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_up_convs_modules_3_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_up_convs_modules_3_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_up_convs_modules_3_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_62 = l_self_modules_decode_head_modules_up_convs_modules_3_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_up_convs_modules_3_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_up_convs_modules_3_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_up_convs_modules_3_modules_1_modules_bn_parameters_bias_ = (None)
        x_64 = torch.nn.functional.relu(x_63, inplace=True)
        x_63 = None
        input_124 = torch.nn.functional.interpolate(
            x_64, [128, 128], None, "bilinear", False
        )
        x_64 = None
        out_28 = torch.cat([input_121, input_122, input_123, input_124], dim=1)
        input_121 = input_122 = input_123 = input_124 = None
        output_48 = torch.conv2d(
            out_28,
            l_self_modules_decode_head_modules_conv_seg_parameters_weight_,
            l_self_modules_decode_head_modules_conv_seg_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_28 = (
            l_self_modules_decode_head_modules_conv_seg_parameters_weight_
        ) = l_self_modules_decode_head_modules_conv_seg_parameters_bias_ = None
        return (output_48,)
