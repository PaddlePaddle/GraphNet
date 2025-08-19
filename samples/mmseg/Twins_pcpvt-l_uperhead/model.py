import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_inputs_: torch.Tensor,
        L_self_modules_backbone_modules_patch_embeds_modules_0_modules_projection_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embeds_modules_0_modules_projection_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embeds_modules_0_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embeds_modules_0_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_position_encodings_modules_0_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_position_encodings_modules_0_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embeds_modules_1_modules_projection_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embeds_modules_1_modules_projection_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embeds_modules_1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embeds_modules_1_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_position_encodings_modules_1_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_position_encodings_modules_1_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_3_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_3_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_4_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_4_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_5_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_5_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_6_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_6_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_6_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_6_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_6_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_7_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_7_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_7_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_7_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_7_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_7_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embeds_modules_2_modules_projection_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embeds_modules_2_modules_projection_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embeds_modules_2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embeds_modules_2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_position_encodings_modules_2_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_position_encodings_modules_2_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_3_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_3_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_5_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_5_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_7_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_7_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_7_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_7_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_7_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_7_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_9_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_9_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_9_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_9_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_9_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_9_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_11_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_11_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_11_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_11_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_11_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_11_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_13_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_13_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_13_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_13_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_13_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_13_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_15_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_15_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_15_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_15_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_15_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_15_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_17_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_17_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_17_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_17_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_17_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_17_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_18_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_18_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_18_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_18_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_18_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_18_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_18_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_18_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_19_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_19_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_19_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_19_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_19_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_19_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_19_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_19_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_20_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_20_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_20_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_20_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_20_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_20_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_20_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_20_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_21_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_21_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_21_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_21_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_21_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_21_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_21_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_21_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_22_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_22_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_22_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_22_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_22_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_22_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_22_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_22_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_23_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_23_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_23_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_23_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_23_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_23_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_23_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_23_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_24_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_24_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_24_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_24_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_24_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_24_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_24_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_24_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_25_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_25_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_25_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_25_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_25_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_25_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_25_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_25_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_26_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_26_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_26_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_26_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_26_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_26_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_26_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_26_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embeds_modules_3_modules_projection_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embeds_modules_3_modules_projection_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embeds_modules_3_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embeds_modules_3_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_position_encodings_modules_3_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_position_encodings_modules_3_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_1_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_1_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_2_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_2_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_backbone_modules_patch_embeds_modules_0_modules_projection_parameters_weight_ = L_self_modules_backbone_modules_patch_embeds_modules_0_modules_projection_parameters_weight_
        l_self_modules_backbone_modules_patch_embeds_modules_0_modules_projection_parameters_bias_ = L_self_modules_backbone_modules_patch_embeds_modules_0_modules_projection_parameters_bias_
        l_self_modules_backbone_modules_patch_embeds_modules_0_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_patch_embeds_modules_0_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_patch_embeds_modules_0_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_patch_embeds_modules_0_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_sr_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_sr_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_sr_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_sr_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_position_encodings_modules_0_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_position_encodings_modules_0_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_position_encodings_modules_0_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_position_encodings_modules_0_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_1_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_1_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_sr_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_sr_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_sr_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_sr_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_1_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_1_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_2_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_2_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_sr_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_sr_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_sr_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_sr_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_2_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_2_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_patch_embeds_modules_1_modules_projection_parameters_weight_ = L_self_modules_backbone_modules_patch_embeds_modules_1_modules_projection_parameters_weight_
        l_self_modules_backbone_modules_patch_embeds_modules_1_modules_projection_parameters_bias_ = L_self_modules_backbone_modules_patch_embeds_modules_1_modules_projection_parameters_bias_
        l_self_modules_backbone_modules_patch_embeds_modules_1_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_patch_embeds_modules_1_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_patch_embeds_modules_1_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_patch_embeds_modules_1_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_sr_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_sr_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_sr_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_sr_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_position_encodings_modules_1_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_position_encodings_modules_1_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_position_encodings_modules_1_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_position_encodings_modules_1_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_1_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_1_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_sr_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_sr_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_sr_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_sr_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_1_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_1_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_2_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_2_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_sr_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_sr_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_sr_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_sr_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_2_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_2_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_3_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_3_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_sr_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_sr_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_sr_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_sr_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_3_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_3_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_3_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_3_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_4_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_4_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_sr_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_sr_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_sr_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_sr_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_4_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_4_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_4_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_4_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_5_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_5_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_sr_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_sr_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_sr_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_sr_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_5_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_5_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_5_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_5_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_6_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_6_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_sr_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_sr_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_sr_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_sr_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_6_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_6_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_6_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_6_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_7_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_7_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_sr_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_sr_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_sr_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_sr_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_7_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_7_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_7_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_7_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_patch_embeds_modules_2_modules_projection_parameters_weight_ = L_self_modules_backbone_modules_patch_embeds_modules_2_modules_projection_parameters_weight_
        l_self_modules_backbone_modules_patch_embeds_modules_2_modules_projection_parameters_bias_ = L_self_modules_backbone_modules_patch_embeds_modules_2_modules_projection_parameters_bias_
        l_self_modules_backbone_modules_patch_embeds_modules_2_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_patch_embeds_modules_2_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_patch_embeds_modules_2_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_patch_embeds_modules_2_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_sr_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_sr_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_sr_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_sr_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_position_encodings_modules_2_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_position_encodings_modules_2_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_position_encodings_modules_2_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_position_encodings_modules_2_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_1_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_1_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_sr_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_sr_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_sr_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_sr_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_1_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_1_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_sr_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_sr_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_sr_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_sr_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_3_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_3_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_sr_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_sr_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_sr_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_sr_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_3_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_3_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_3_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_3_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_sr_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_sr_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_sr_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_sr_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_5_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_5_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_sr_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_sr_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_sr_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_sr_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_5_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_5_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_5_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_5_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_sr_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_sr_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_sr_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_sr_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_7_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_7_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_sr_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_sr_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_sr_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_sr_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_7_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_7_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_7_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_7_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_sr_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_sr_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_sr_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_sr_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_9_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_9_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_sr_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_sr_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_sr_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_sr_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_9_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_9_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_9_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_9_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_sr_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_sr_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_sr_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_sr_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_11_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_11_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_sr_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_sr_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_sr_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_sr_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_11_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_11_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_11_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_11_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_sr_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_sr_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_sr_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_sr_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_13_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_13_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_sr_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_sr_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_sr_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_sr_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_13_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_13_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_13_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_13_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_sr_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_sr_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_sr_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_sr_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_15_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_15_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_sr_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_sr_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_sr_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_sr_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_15_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_15_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_15_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_15_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_sr_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_sr_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_sr_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_sr_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_17_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_17_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_sr_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_sr_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_sr_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_sr_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_17_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_17_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_17_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_17_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_18_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_18_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_sr_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_sr_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_sr_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_sr_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_18_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_18_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_18_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_18_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_18_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_18_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_19_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_19_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_sr_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_sr_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_sr_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_sr_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_19_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_19_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_19_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_19_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_19_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_19_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_20_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_20_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_sr_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_sr_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_sr_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_sr_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_20_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_20_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_20_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_20_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_20_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_20_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_21_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_21_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_sr_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_sr_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_sr_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_sr_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_21_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_21_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_21_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_21_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_21_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_21_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_22_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_22_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_sr_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_sr_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_sr_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_sr_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_22_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_22_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_22_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_22_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_22_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_22_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_23_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_23_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_sr_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_sr_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_sr_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_sr_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_23_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_23_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_23_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_23_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_23_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_23_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_24_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_24_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_sr_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_sr_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_sr_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_sr_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_24_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_24_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_24_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_24_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_24_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_24_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_25_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_25_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_sr_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_sr_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_sr_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_sr_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_25_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_25_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_25_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_25_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_25_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_25_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_26_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_26_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_sr_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_sr_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_sr_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_sr_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_26_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_26_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_26_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_26_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_26_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_26_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_patch_embeds_modules_3_modules_projection_parameters_weight_ = L_self_modules_backbone_modules_patch_embeds_modules_3_modules_projection_parameters_weight_
        l_self_modules_backbone_modules_patch_embeds_modules_3_modules_projection_parameters_bias_ = L_self_modules_backbone_modules_patch_embeds_modules_3_modules_projection_parameters_bias_
        l_self_modules_backbone_modules_patch_embeds_modules_3_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_patch_embeds_modules_3_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_patch_embeds_modules_3_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_patch_embeds_modules_3_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_position_encodings_modules_3_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_position_encodings_modules_3_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_position_encodings_modules_3_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_position_encodings_modules_3_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_1_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_1_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_1_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_1_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_1_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_1_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_2_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_2_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_2_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_2_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_2_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_2_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_
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
            l_self_modules_backbone_modules_patch_embeds_modules_0_modules_projection_parameters_weight_,
            l_self_modules_backbone_modules_patch_embeds_modules_0_modules_projection_parameters_bias_,
            (4, 4),
            (0, 0),
            (1, 1),
            1,
        )
        l_inputs_ = l_self_modules_backbone_modules_patch_embeds_modules_0_modules_projection_parameters_weight_ = l_self_modules_backbone_modules_patch_embeds_modules_0_modules_projection_parameters_bias_ = (None)
        flatten = x.flatten(2)
        x = None
        x_1 = flatten.transpose(1, 2)
        flatten = None
        x_2 = torch.nn.functional.layer_norm(
            x_1,
            (64,),
            l_self_modules_backbone_modules_patch_embeds_modules_0_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_patch_embeds_modules_0_modules_norm_parameters_bias_,
            1e-05,
        )
        x_1 = l_self_modules_backbone_modules_patch_embeds_modules_0_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_patch_embeds_modules_0_modules_norm_parameters_bias_ = (None)
        x_3 = torch.nn.functional.dropout(x_2, 0.0, False, False)
        x_2 = None
        x_q = torch.nn.functional.layer_norm(
            x_3,
            (64,),
            l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_norm1_parameters_bias_ = (None)
        transpose_1 = x_q.transpose(1, 2)
        x_kv = transpose_1.reshape(1, 64, 128, 128)
        transpose_1 = None
        x_kv_1 = torch.conv2d(
            x_kv,
            l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_sr_parameters_bias_,
            (8, 8),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv = l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_1 = x_kv_1.flatten(2)
        x_kv_1 = None
        transpose_2 = flatten_1.transpose(1, 2)
        flatten_1 = None
        x_kv_2 = transpose_2.contiguous()
        transpose_2 = None
        x_kv_3 = torch.nn.functional.layer_norm(
            x_kv_2,
            (64,),
            l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_2 = l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_1 = x_q.transpose(0, 1)
        x_q = None
        x_kv_4 = x_kv_3.transpose(0, 1)
        x_kv_3 = None
        multi_head_attention_forward = torch.nn.functional.multi_head_attention_forward(
            x_q_1,
            x_kv_4,
            x_kv_4,
            64,
            1,
            l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_1 = (
            x_kv_4
        ) = l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output = multi_head_attention_forward[0]
        multi_head_attention_forward = None
        out = attn_output.transpose(0, 1)
        attn_output = None
        dropout_1 = torch.nn.functional.dropout(out, 0.0, False, False)
        out = None
        add = 0.0 + dropout_1
        dropout_1 = None
        x_4 = x_3 + add
        x_3 = add = None
        layer_norm_3 = torch.nn.functional.layer_norm(
            x_4,
            (64,),
            l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_norm2_parameters_bias_ = (None)
        input_1 = torch._C._nn.linear(
            layer_norm_3,
            l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_3 = l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_2 = torch._C._nn.gelu(input_1, approximate="none")
        input_1 = None
        input_3 = torch.nn.functional.dropout(input_2, 0.0, False, False)
        input_2 = None
        input_4 = torch._C._nn.linear(
            input_3,
            l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_3 = l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_5 = torch.nn.functional.dropout(input_4, 0.0, False, False)
        input_4 = None
        x_5 = x_4 + input_5
        x_4 = input_5 = None
        transpose_6 = x_5.transpose(1, 2)
        x_5 = None
        cnn_feat = transpose_6.view(1, 64, 128, 128)
        transpose_6 = None
        conv2d_2 = torch.conv2d(
            cnn_feat,
            l_self_modules_backbone_modules_position_encodings_modules_0_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_position_encodings_modules_0_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            64,
        )
        l_self_modules_backbone_modules_position_encodings_modules_0_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_position_encodings_modules_0_modules_proj_parameters_bias_ = (None)
        x_6 = conv2d_2 + cnn_feat
        conv2d_2 = cnn_feat = None
        flatten_2 = x_6.flatten(2)
        x_6 = None
        x_7 = flatten_2.transpose(1, 2)
        flatten_2 = None
        x_q_2 = torch.nn.functional.layer_norm(
            x_7,
            (64,),
            l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_norm1_parameters_bias_ = (None)
        transpose_8 = x_q_2.transpose(1, 2)
        x_kv_5 = transpose_8.reshape(1, 64, 128, 128)
        transpose_8 = None
        x_kv_6 = torch.conv2d(
            x_kv_5,
            l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_sr_parameters_bias_,
            (8, 8),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_5 = l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_3 = x_kv_6.flatten(2)
        x_kv_6 = None
        transpose_9 = flatten_3.transpose(1, 2)
        flatten_3 = None
        x_kv_7 = transpose_9.contiguous()
        transpose_9 = None
        x_kv_8 = torch.nn.functional.layer_norm(
            x_kv_7,
            (64,),
            l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_7 = l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_3 = x_q_2.transpose(0, 1)
        x_q_2 = None
        x_kv_9 = x_kv_8.transpose(0, 1)
        x_kv_8 = None
        multi_head_attention_forward_1 = torch.nn.functional.multi_head_attention_forward(
            x_q_3,
            x_kv_9,
            x_kv_9,
            64,
            1,
            l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_3 = (
            x_kv_9
        ) = l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_1 = multi_head_attention_forward_1[0]
        multi_head_attention_forward_1 = None
        out_1 = attn_output_1.transpose(0, 1)
        attn_output_1 = None
        dropout_4 = torch.nn.functional.dropout(out_1, 0.0, False, False)
        out_1 = None
        add_4 = 0.0 + dropout_4
        dropout_4 = None
        x_8 = x_7 + add_4
        x_7 = add_4 = None
        layer_norm_6 = torch.nn.functional.layer_norm(
            x_8,
            (64,),
            l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_norm2_parameters_bias_ = (None)
        input_6 = torch._C._nn.linear(
            layer_norm_6,
            l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_6 = l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_7 = torch._C._nn.gelu(input_6, approximate="none")
        input_6 = None
        input_8 = torch.nn.functional.dropout(input_7, 0.0, False, False)
        input_7 = None
        input_9 = torch._C._nn.linear(
            input_8,
            l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_8 = l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_10 = torch.nn.functional.dropout(input_9, 0.0, False, False)
        input_9 = None
        x_9 = x_8 + input_10
        x_8 = input_10 = None
        x_q_4 = torch.nn.functional.layer_norm(
            x_9,
            (64,),
            l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_norm1_parameters_bias_ = (None)
        transpose_13 = x_q_4.transpose(1, 2)
        x_kv_10 = transpose_13.reshape(1, 64, 128, 128)
        transpose_13 = None
        x_kv_11 = torch.conv2d(
            x_kv_10,
            l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_sr_parameters_bias_,
            (8, 8),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_10 = l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_4 = x_kv_11.flatten(2)
        x_kv_11 = None
        transpose_14 = flatten_4.transpose(1, 2)
        flatten_4 = None
        x_kv_12 = transpose_14.contiguous()
        transpose_14 = None
        x_kv_13 = torch.nn.functional.layer_norm(
            x_kv_12,
            (64,),
            l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_12 = l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_5 = x_q_4.transpose(0, 1)
        x_q_4 = None
        x_kv_14 = x_kv_13.transpose(0, 1)
        x_kv_13 = None
        multi_head_attention_forward_2 = torch.nn.functional.multi_head_attention_forward(
            x_q_5,
            x_kv_14,
            x_kv_14,
            64,
            1,
            l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_5 = (
            x_kv_14
        ) = l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_2 = multi_head_attention_forward_2[0]
        multi_head_attention_forward_2 = None
        out_2 = attn_output_2.transpose(0, 1)
        attn_output_2 = None
        dropout_7 = torch.nn.functional.dropout(out_2, 0.0, False, False)
        out_2 = None
        add_7 = 0.0 + dropout_7
        dropout_7 = None
        x_10 = x_9 + add_7
        x_9 = add_7 = None
        layer_norm_9 = torch.nn.functional.layer_norm(
            x_10,
            (64,),
            l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_norm2_parameters_bias_ = (None)
        input_11 = torch._C._nn.linear(
            layer_norm_9,
            l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_9 = l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_12 = torch._C._nn.gelu(input_11, approximate="none")
        input_11 = None
        input_13 = torch.nn.functional.dropout(input_12, 0.0, False, False)
        input_12 = None
        input_14 = torch._C._nn.linear(
            input_13,
            l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_13 = l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_15 = torch.nn.functional.dropout(input_14, 0.0, False, False)
        input_14 = None
        x_11 = x_10 + input_15
        x_10 = input_15 = None
        reshape_3 = x_11.reshape(1, 128, 128, -1)
        x_11 = None
        permute = reshape_3.permute(0, 3, 1, 2)
        reshape_3 = None
        x_12 = permute.contiguous()
        permute = None
        x_13 = torch.conv2d(
            x_12,
            l_self_modules_backbone_modules_patch_embeds_modules_1_modules_projection_parameters_weight_,
            l_self_modules_backbone_modules_patch_embeds_modules_1_modules_projection_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_patch_embeds_modules_1_modules_projection_parameters_weight_ = l_self_modules_backbone_modules_patch_embeds_modules_1_modules_projection_parameters_bias_ = (None)
        flatten_5 = x_13.flatten(2)
        x_13 = None
        x_14 = flatten_5.transpose(1, 2)
        flatten_5 = None
        x_15 = torch.nn.functional.layer_norm(
            x_14,
            (128,),
            l_self_modules_backbone_modules_patch_embeds_modules_1_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_patch_embeds_modules_1_modules_norm_parameters_bias_,
            1e-05,
        )
        x_14 = l_self_modules_backbone_modules_patch_embeds_modules_1_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_patch_embeds_modules_1_modules_norm_parameters_bias_ = (None)
        x_16 = torch.nn.functional.dropout(x_15, 0.0, False, False)
        x_15 = None
        x_q_6 = torch.nn.functional.layer_norm(
            x_16,
            (128,),
            l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_norm1_parameters_bias_ = (None)
        transpose_19 = x_q_6.transpose(1, 2)
        x_kv_15 = transpose_19.reshape(1, 128, 64, 64)
        transpose_19 = None
        x_kv_16 = torch.conv2d(
            x_kv_15,
            l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_sr_parameters_bias_,
            (4, 4),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_15 = l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_6 = x_kv_16.flatten(2)
        x_kv_16 = None
        transpose_20 = flatten_6.transpose(1, 2)
        flatten_6 = None
        x_kv_17 = transpose_20.contiguous()
        transpose_20 = None
        x_kv_18 = torch.nn.functional.layer_norm(
            x_kv_17,
            (128,),
            l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_17 = l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_7 = x_q_6.transpose(0, 1)
        x_q_6 = None
        x_kv_19 = x_kv_18.transpose(0, 1)
        x_kv_18 = None
        multi_head_attention_forward_3 = torch.nn.functional.multi_head_attention_forward(
            x_q_7,
            x_kv_19,
            x_kv_19,
            128,
            2,
            l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_7 = (
            x_kv_19
        ) = l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_3 = multi_head_attention_forward_3[0]
        multi_head_attention_forward_3 = None
        out_3 = attn_output_3.transpose(0, 1)
        attn_output_3 = None
        dropout_11 = torch.nn.functional.dropout(out_3, 0.0, False, False)
        out_3 = None
        add_10 = 0.0 + dropout_11
        dropout_11 = None
        x_17 = x_16 + add_10
        x_16 = add_10 = None
        layer_norm_13 = torch.nn.functional.layer_norm(
            x_17,
            (128,),
            l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_norm2_parameters_bias_ = (None)
        input_16 = torch._C._nn.linear(
            layer_norm_13,
            l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_13 = l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_17 = torch._C._nn.gelu(input_16, approximate="none")
        input_16 = None
        input_18 = torch.nn.functional.dropout(input_17, 0.0, False, False)
        input_17 = None
        input_19 = torch._C._nn.linear(
            input_18,
            l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_18 = l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_20 = torch.nn.functional.dropout(input_19, 0.0, False, False)
        input_19 = None
        x_18 = x_17 + input_20
        x_17 = input_20 = None
        transpose_24 = x_18.transpose(1, 2)
        x_18 = None
        cnn_feat_1 = transpose_24.view(1, 128, 64, 64)
        transpose_24 = None
        conv2d_7 = torch.conv2d(
            cnn_feat_1,
            l_self_modules_backbone_modules_position_encodings_modules_1_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_position_encodings_modules_1_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        l_self_modules_backbone_modules_position_encodings_modules_1_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_position_encodings_modules_1_modules_proj_parameters_bias_ = (None)
        x_19 = conv2d_7 + cnn_feat_1
        conv2d_7 = cnn_feat_1 = None
        flatten_7 = x_19.flatten(2)
        x_19 = None
        x_20 = flatten_7.transpose(1, 2)
        flatten_7 = None
        x_q_8 = torch.nn.functional.layer_norm(
            x_20,
            (128,),
            l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_norm1_parameters_bias_ = (None)
        transpose_26 = x_q_8.transpose(1, 2)
        x_kv_20 = transpose_26.reshape(1, 128, 64, 64)
        transpose_26 = None
        x_kv_21 = torch.conv2d(
            x_kv_20,
            l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_sr_parameters_bias_,
            (4, 4),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_20 = l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_8 = x_kv_21.flatten(2)
        x_kv_21 = None
        transpose_27 = flatten_8.transpose(1, 2)
        flatten_8 = None
        x_kv_22 = transpose_27.contiguous()
        transpose_27 = None
        x_kv_23 = torch.nn.functional.layer_norm(
            x_kv_22,
            (128,),
            l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_22 = l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_9 = x_q_8.transpose(0, 1)
        x_q_8 = None
        x_kv_24 = x_kv_23.transpose(0, 1)
        x_kv_23 = None
        multi_head_attention_forward_4 = torch.nn.functional.multi_head_attention_forward(
            x_q_9,
            x_kv_24,
            x_kv_24,
            128,
            2,
            l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_9 = (
            x_kv_24
        ) = l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_4 = multi_head_attention_forward_4[0]
        multi_head_attention_forward_4 = None
        out_4 = attn_output_4.transpose(0, 1)
        attn_output_4 = None
        dropout_14 = torch.nn.functional.dropout(out_4, 0.0, False, False)
        out_4 = None
        add_14 = 0.0 + dropout_14
        dropout_14 = None
        x_21 = x_20 + add_14
        x_20 = add_14 = None
        layer_norm_16 = torch.nn.functional.layer_norm(
            x_21,
            (128,),
            l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_norm2_parameters_bias_ = (None)
        input_21 = torch._C._nn.linear(
            layer_norm_16,
            l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_16 = l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_22 = torch._C._nn.gelu(input_21, approximate="none")
        input_21 = None
        input_23 = torch.nn.functional.dropout(input_22, 0.0, False, False)
        input_22 = None
        input_24 = torch._C._nn.linear(
            input_23,
            l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_23 = l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_25 = torch.nn.functional.dropout(input_24, 0.0, False, False)
        input_24 = None
        x_22 = x_21 + input_25
        x_21 = input_25 = None
        x_q_10 = torch.nn.functional.layer_norm(
            x_22,
            (128,),
            l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_norm1_parameters_bias_ = (None)
        transpose_31 = x_q_10.transpose(1, 2)
        x_kv_25 = transpose_31.reshape(1, 128, 64, 64)
        transpose_31 = None
        x_kv_26 = torch.conv2d(
            x_kv_25,
            l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_sr_parameters_bias_,
            (4, 4),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_25 = l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_9 = x_kv_26.flatten(2)
        x_kv_26 = None
        transpose_32 = flatten_9.transpose(1, 2)
        flatten_9 = None
        x_kv_27 = transpose_32.contiguous()
        transpose_32 = None
        x_kv_28 = torch.nn.functional.layer_norm(
            x_kv_27,
            (128,),
            l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_27 = l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_11 = x_q_10.transpose(0, 1)
        x_q_10 = None
        x_kv_29 = x_kv_28.transpose(0, 1)
        x_kv_28 = None
        multi_head_attention_forward_5 = torch.nn.functional.multi_head_attention_forward(
            x_q_11,
            x_kv_29,
            x_kv_29,
            128,
            2,
            l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_11 = (
            x_kv_29
        ) = l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_5 = multi_head_attention_forward_5[0]
        multi_head_attention_forward_5 = None
        out_5 = attn_output_5.transpose(0, 1)
        attn_output_5 = None
        dropout_17 = torch.nn.functional.dropout(out_5, 0.0, False, False)
        out_5 = None
        add_17 = 0.0 + dropout_17
        dropout_17 = None
        x_23 = x_22 + add_17
        x_22 = add_17 = None
        layer_norm_19 = torch.nn.functional.layer_norm(
            x_23,
            (128,),
            l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_norm2_parameters_bias_ = (None)
        input_26 = torch._C._nn.linear(
            layer_norm_19,
            l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_19 = l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_27 = torch._C._nn.gelu(input_26, approximate="none")
        input_26 = None
        input_28 = torch.nn.functional.dropout(input_27, 0.0, False, False)
        input_27 = None
        input_29 = torch._C._nn.linear(
            input_28,
            l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_28 = l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_30 = torch.nn.functional.dropout(input_29, 0.0, False, False)
        input_29 = None
        x_24 = x_23 + input_30
        x_23 = input_30 = None
        x_q_12 = torch.nn.functional.layer_norm(
            x_24,
            (128,),
            l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_norm1_parameters_bias_ = (None)
        transpose_36 = x_q_12.transpose(1, 2)
        x_kv_30 = transpose_36.reshape(1, 128, 64, 64)
        transpose_36 = None
        x_kv_31 = torch.conv2d(
            x_kv_30,
            l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_sr_parameters_bias_,
            (4, 4),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_30 = l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_10 = x_kv_31.flatten(2)
        x_kv_31 = None
        transpose_37 = flatten_10.transpose(1, 2)
        flatten_10 = None
        x_kv_32 = transpose_37.contiguous()
        transpose_37 = None
        x_kv_33 = torch.nn.functional.layer_norm(
            x_kv_32,
            (128,),
            l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_32 = l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_13 = x_q_12.transpose(0, 1)
        x_q_12 = None
        x_kv_34 = x_kv_33.transpose(0, 1)
        x_kv_33 = None
        multi_head_attention_forward_6 = torch.nn.functional.multi_head_attention_forward(
            x_q_13,
            x_kv_34,
            x_kv_34,
            128,
            2,
            l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_13 = (
            x_kv_34
        ) = l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_6 = multi_head_attention_forward_6[0]
        multi_head_attention_forward_6 = None
        out_6 = attn_output_6.transpose(0, 1)
        attn_output_6 = None
        dropout_20 = torch.nn.functional.dropout(out_6, 0.0, False, False)
        out_6 = None
        add_20 = 0.0 + dropout_20
        dropout_20 = None
        x_25 = x_24 + add_20
        x_24 = add_20 = None
        layer_norm_22 = torch.nn.functional.layer_norm(
            x_25,
            (128,),
            l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_norm2_parameters_bias_ = (None)
        input_31 = torch._C._nn.linear(
            layer_norm_22,
            l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_22 = l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_32 = torch._C._nn.gelu(input_31, approximate="none")
        input_31 = None
        input_33 = torch.nn.functional.dropout(input_32, 0.0, False, False)
        input_32 = None
        input_34 = torch._C._nn.linear(
            input_33,
            l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_33 = l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_3_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_35 = torch.nn.functional.dropout(input_34, 0.0, False, False)
        input_34 = None
        x_26 = x_25 + input_35
        x_25 = input_35 = None
        x_q_14 = torch.nn.functional.layer_norm(
            x_26,
            (128,),
            l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_norm1_parameters_bias_ = (None)
        transpose_41 = x_q_14.transpose(1, 2)
        x_kv_35 = transpose_41.reshape(1, 128, 64, 64)
        transpose_41 = None
        x_kv_36 = torch.conv2d(
            x_kv_35,
            l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_sr_parameters_bias_,
            (4, 4),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_35 = l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_11 = x_kv_36.flatten(2)
        x_kv_36 = None
        transpose_42 = flatten_11.transpose(1, 2)
        flatten_11 = None
        x_kv_37 = transpose_42.contiguous()
        transpose_42 = None
        x_kv_38 = torch.nn.functional.layer_norm(
            x_kv_37,
            (128,),
            l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_37 = l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_15 = x_q_14.transpose(0, 1)
        x_q_14 = None
        x_kv_39 = x_kv_38.transpose(0, 1)
        x_kv_38 = None
        multi_head_attention_forward_7 = torch.nn.functional.multi_head_attention_forward(
            x_q_15,
            x_kv_39,
            x_kv_39,
            128,
            2,
            l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_15 = (
            x_kv_39
        ) = l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_7 = multi_head_attention_forward_7[0]
        multi_head_attention_forward_7 = None
        out_7 = attn_output_7.transpose(0, 1)
        attn_output_7 = None
        dropout_23 = torch.nn.functional.dropout(out_7, 0.0, False, False)
        out_7 = None
        add_23 = 0.0 + dropout_23
        dropout_23 = None
        x_27 = x_26 + add_23
        x_26 = add_23 = None
        layer_norm_25 = torch.nn.functional.layer_norm(
            x_27,
            (128,),
            l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_norm2_parameters_bias_ = (None)
        input_36 = torch._C._nn.linear(
            layer_norm_25,
            l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_25 = l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_37 = torch._C._nn.gelu(input_36, approximate="none")
        input_36 = None
        input_38 = torch.nn.functional.dropout(input_37, 0.0, False, False)
        input_37 = None
        input_39 = torch._C._nn.linear(
            input_38,
            l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_38 = l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_4_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_40 = torch.nn.functional.dropout(input_39, 0.0, False, False)
        input_39 = None
        x_28 = x_27 + input_40
        x_27 = input_40 = None
        x_q_16 = torch.nn.functional.layer_norm(
            x_28,
            (128,),
            l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_norm1_parameters_bias_ = (None)
        transpose_46 = x_q_16.transpose(1, 2)
        x_kv_40 = transpose_46.reshape(1, 128, 64, 64)
        transpose_46 = None
        x_kv_41 = torch.conv2d(
            x_kv_40,
            l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_sr_parameters_bias_,
            (4, 4),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_40 = l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_12 = x_kv_41.flatten(2)
        x_kv_41 = None
        transpose_47 = flatten_12.transpose(1, 2)
        flatten_12 = None
        x_kv_42 = transpose_47.contiguous()
        transpose_47 = None
        x_kv_43 = torch.nn.functional.layer_norm(
            x_kv_42,
            (128,),
            l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_42 = l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_17 = x_q_16.transpose(0, 1)
        x_q_16 = None
        x_kv_44 = x_kv_43.transpose(0, 1)
        x_kv_43 = None
        multi_head_attention_forward_8 = torch.nn.functional.multi_head_attention_forward(
            x_q_17,
            x_kv_44,
            x_kv_44,
            128,
            2,
            l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_17 = (
            x_kv_44
        ) = l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_8 = multi_head_attention_forward_8[0]
        multi_head_attention_forward_8 = None
        out_8 = attn_output_8.transpose(0, 1)
        attn_output_8 = None
        dropout_26 = torch.nn.functional.dropout(out_8, 0.0, False, False)
        out_8 = None
        add_26 = 0.0 + dropout_26
        dropout_26 = None
        x_29 = x_28 + add_26
        x_28 = add_26 = None
        layer_norm_28 = torch.nn.functional.layer_norm(
            x_29,
            (128,),
            l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_norm2_parameters_bias_ = (None)
        input_41 = torch._C._nn.linear(
            layer_norm_28,
            l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_28 = l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_42 = torch._C._nn.gelu(input_41, approximate="none")
        input_41 = None
        input_43 = torch.nn.functional.dropout(input_42, 0.0, False, False)
        input_42 = None
        input_44 = torch._C._nn.linear(
            input_43,
            l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_43 = l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_5_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_45 = torch.nn.functional.dropout(input_44, 0.0, False, False)
        input_44 = None
        x_30 = x_29 + input_45
        x_29 = input_45 = None
        x_q_18 = torch.nn.functional.layer_norm(
            x_30,
            (128,),
            l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_norm1_parameters_bias_ = (None)
        transpose_51 = x_q_18.transpose(1, 2)
        x_kv_45 = transpose_51.reshape(1, 128, 64, 64)
        transpose_51 = None
        x_kv_46 = torch.conv2d(
            x_kv_45,
            l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_sr_parameters_bias_,
            (4, 4),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_45 = l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_13 = x_kv_46.flatten(2)
        x_kv_46 = None
        transpose_52 = flatten_13.transpose(1, 2)
        flatten_13 = None
        x_kv_47 = transpose_52.contiguous()
        transpose_52 = None
        x_kv_48 = torch.nn.functional.layer_norm(
            x_kv_47,
            (128,),
            l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_47 = l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_19 = x_q_18.transpose(0, 1)
        x_q_18 = None
        x_kv_49 = x_kv_48.transpose(0, 1)
        x_kv_48 = None
        multi_head_attention_forward_9 = torch.nn.functional.multi_head_attention_forward(
            x_q_19,
            x_kv_49,
            x_kv_49,
            128,
            2,
            l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_19 = (
            x_kv_49
        ) = l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_9 = multi_head_attention_forward_9[0]
        multi_head_attention_forward_9 = None
        out_9 = attn_output_9.transpose(0, 1)
        attn_output_9 = None
        dropout_29 = torch.nn.functional.dropout(out_9, 0.0, False, False)
        out_9 = None
        add_29 = 0.0 + dropout_29
        dropout_29 = None
        x_31 = x_30 + add_29
        x_30 = add_29 = None
        layer_norm_31 = torch.nn.functional.layer_norm(
            x_31,
            (128,),
            l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_norm2_parameters_bias_ = (None)
        input_46 = torch._C._nn.linear(
            layer_norm_31,
            l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_31 = l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_47 = torch._C._nn.gelu(input_46, approximate="none")
        input_46 = None
        input_48 = torch.nn.functional.dropout(input_47, 0.0, False, False)
        input_47 = None
        input_49 = torch._C._nn.linear(
            input_48,
            l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_48 = l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_6_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_50 = torch.nn.functional.dropout(input_49, 0.0, False, False)
        input_49 = None
        x_32 = x_31 + input_50
        x_31 = input_50 = None
        x_q_20 = torch.nn.functional.layer_norm(
            x_32,
            (128,),
            l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_norm1_parameters_bias_ = (None)
        transpose_56 = x_q_20.transpose(1, 2)
        x_kv_50 = transpose_56.reshape(1, 128, 64, 64)
        transpose_56 = None
        x_kv_51 = torch.conv2d(
            x_kv_50,
            l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_sr_parameters_bias_,
            (4, 4),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_50 = l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_14 = x_kv_51.flatten(2)
        x_kv_51 = None
        transpose_57 = flatten_14.transpose(1, 2)
        flatten_14 = None
        x_kv_52 = transpose_57.contiguous()
        transpose_57 = None
        x_kv_53 = torch.nn.functional.layer_norm(
            x_kv_52,
            (128,),
            l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_52 = l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_21 = x_q_20.transpose(0, 1)
        x_q_20 = None
        x_kv_54 = x_kv_53.transpose(0, 1)
        x_kv_53 = None
        multi_head_attention_forward_10 = torch.nn.functional.multi_head_attention_forward(
            x_q_21,
            x_kv_54,
            x_kv_54,
            128,
            2,
            l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_21 = (
            x_kv_54
        ) = l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_10 = multi_head_attention_forward_10[0]
        multi_head_attention_forward_10 = None
        out_10 = attn_output_10.transpose(0, 1)
        attn_output_10 = None
        dropout_32 = torch.nn.functional.dropout(out_10, 0.0, False, False)
        out_10 = None
        add_32 = 0.0 + dropout_32
        dropout_32 = None
        x_33 = x_32 + add_32
        x_32 = add_32 = None
        layer_norm_34 = torch.nn.functional.layer_norm(
            x_33,
            (128,),
            l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_norm2_parameters_bias_ = (None)
        input_51 = torch._C._nn.linear(
            layer_norm_34,
            l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_34 = l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_52 = torch._C._nn.gelu(input_51, approximate="none")
        input_51 = None
        input_53 = torch.nn.functional.dropout(input_52, 0.0, False, False)
        input_52 = None
        input_54 = torch._C._nn.linear(
            input_53,
            l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_53 = l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_7_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_55 = torch.nn.functional.dropout(input_54, 0.0, False, False)
        input_54 = None
        x_34 = x_33 + input_55
        x_33 = input_55 = None
        reshape_12 = x_34.reshape(1, 64, 64, -1)
        x_34 = None
        permute_1 = reshape_12.permute(0, 3, 1, 2)
        reshape_12 = None
        x_35 = permute_1.contiguous()
        permute_1 = None
        x_36 = torch.conv2d(
            x_35,
            l_self_modules_backbone_modules_patch_embeds_modules_2_modules_projection_parameters_weight_,
            l_self_modules_backbone_modules_patch_embeds_modules_2_modules_projection_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_patch_embeds_modules_2_modules_projection_parameters_weight_ = l_self_modules_backbone_modules_patch_embeds_modules_2_modules_projection_parameters_bias_ = (None)
        flatten_15 = x_36.flatten(2)
        x_36 = None
        x_37 = flatten_15.transpose(1, 2)
        flatten_15 = None
        x_38 = torch.nn.functional.layer_norm(
            x_37,
            (320,),
            l_self_modules_backbone_modules_patch_embeds_modules_2_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_patch_embeds_modules_2_modules_norm_parameters_bias_,
            1e-05,
        )
        x_37 = l_self_modules_backbone_modules_patch_embeds_modules_2_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_patch_embeds_modules_2_modules_norm_parameters_bias_ = (None)
        x_39 = torch.nn.functional.dropout(x_38, 0.0, False, False)
        x_38 = None
        x_q_22 = torch.nn.functional.layer_norm(
            x_39,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm1_parameters_bias_ = (None)
        transpose_62 = x_q_22.transpose(1, 2)
        x_kv_55 = transpose_62.reshape(1, 320, 32, 32)
        transpose_62 = None
        x_kv_56 = torch.conv2d(
            x_kv_55,
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_55 = l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_16 = x_kv_56.flatten(2)
        x_kv_56 = None
        transpose_63 = flatten_16.transpose(1, 2)
        flatten_16 = None
        x_kv_57 = transpose_63.contiguous()
        transpose_63 = None
        x_kv_58 = torch.nn.functional.layer_norm(
            x_kv_57,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_57 = l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_23 = x_q_22.transpose(0, 1)
        x_q_22 = None
        x_kv_59 = x_kv_58.transpose(0, 1)
        x_kv_58 = None
        multi_head_attention_forward_11 = torch.nn.functional.multi_head_attention_forward(
            x_q_23,
            x_kv_59,
            x_kv_59,
            320,
            5,
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_23 = (
            x_kv_59
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_11 = multi_head_attention_forward_11[0]
        multi_head_attention_forward_11 = None
        out_11 = attn_output_11.transpose(0, 1)
        attn_output_11 = None
        dropout_36 = torch.nn.functional.dropout(out_11, 0.0, False, False)
        out_11 = None
        add_35 = 0.0 + dropout_36
        dropout_36 = None
        x_40 = x_39 + add_35
        x_39 = add_35 = None
        layer_norm_38 = torch.nn.functional.layer_norm(
            x_40,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm2_parameters_bias_ = (None)
        input_56 = torch._C._nn.linear(
            layer_norm_38,
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_38 = l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_57 = torch._C._nn.gelu(input_56, approximate="none")
        input_56 = None
        input_58 = torch.nn.functional.dropout(input_57, 0.0, False, False)
        input_57 = None
        input_59 = torch._C._nn.linear(
            input_58,
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_58 = l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_60 = torch.nn.functional.dropout(input_59, 0.0, False, False)
        input_59 = None
        x_41 = x_40 + input_60
        x_40 = input_60 = None
        transpose_67 = x_41.transpose(1, 2)
        x_41 = None
        cnn_feat_2 = transpose_67.view(1, 320, 32, 32)
        transpose_67 = None
        conv2d_17 = torch.conv2d(
            cnn_feat_2,
            l_self_modules_backbone_modules_position_encodings_modules_2_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_position_encodings_modules_2_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_position_encodings_modules_2_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_position_encodings_modules_2_modules_proj_parameters_bias_ = (None)
        x_42 = conv2d_17 + cnn_feat_2
        conv2d_17 = cnn_feat_2 = None
        flatten_17 = x_42.flatten(2)
        x_42 = None
        x_43 = flatten_17.transpose(1, 2)
        flatten_17 = None
        x_q_24 = torch.nn.functional.layer_norm(
            x_43,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_norm1_parameters_bias_ = (None)
        transpose_69 = x_q_24.transpose(1, 2)
        x_kv_60 = transpose_69.reshape(1, 320, 32, 32)
        transpose_69 = None
        x_kv_61 = torch.conv2d(
            x_kv_60,
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_60 = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_18 = x_kv_61.flatten(2)
        x_kv_61 = None
        transpose_70 = flatten_18.transpose(1, 2)
        flatten_18 = None
        x_kv_62 = transpose_70.contiguous()
        transpose_70 = None
        x_kv_63 = torch.nn.functional.layer_norm(
            x_kv_62,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_62 = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_25 = x_q_24.transpose(0, 1)
        x_q_24 = None
        x_kv_64 = x_kv_63.transpose(0, 1)
        x_kv_63 = None
        multi_head_attention_forward_12 = torch.nn.functional.multi_head_attention_forward(
            x_q_25,
            x_kv_64,
            x_kv_64,
            320,
            5,
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_25 = (
            x_kv_64
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_12 = multi_head_attention_forward_12[0]
        multi_head_attention_forward_12 = None
        out_12 = attn_output_12.transpose(0, 1)
        attn_output_12 = None
        dropout_39 = torch.nn.functional.dropout(out_12, 0.0, False, False)
        out_12 = None
        add_39 = 0.0 + dropout_39
        dropout_39 = None
        x_44 = x_43 + add_39
        x_43 = add_39 = None
        layer_norm_41 = torch.nn.functional.layer_norm(
            x_44,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_norm2_parameters_bias_ = (None)
        input_61 = torch._C._nn.linear(
            layer_norm_41,
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_41 = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_62 = torch._C._nn.gelu(input_61, approximate="none")
        input_61 = None
        input_63 = torch.nn.functional.dropout(input_62, 0.0, False, False)
        input_62 = None
        input_64 = torch._C._nn.linear(
            input_63,
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_63 = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_65 = torch.nn.functional.dropout(input_64, 0.0, False, False)
        input_64 = None
        x_45 = x_44 + input_65
        x_44 = input_65 = None
        x_q_26 = torch.nn.functional.layer_norm(
            x_45,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_norm1_parameters_bias_ = (None)
        transpose_74 = x_q_26.transpose(1, 2)
        x_kv_65 = transpose_74.reshape(1, 320, 32, 32)
        transpose_74 = None
        x_kv_66 = torch.conv2d(
            x_kv_65,
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_65 = l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_19 = x_kv_66.flatten(2)
        x_kv_66 = None
        transpose_75 = flatten_19.transpose(1, 2)
        flatten_19 = None
        x_kv_67 = transpose_75.contiguous()
        transpose_75 = None
        x_kv_68 = torch.nn.functional.layer_norm(
            x_kv_67,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_67 = l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_27 = x_q_26.transpose(0, 1)
        x_q_26 = None
        x_kv_69 = x_kv_68.transpose(0, 1)
        x_kv_68 = None
        multi_head_attention_forward_13 = torch.nn.functional.multi_head_attention_forward(
            x_q_27,
            x_kv_69,
            x_kv_69,
            320,
            5,
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_27 = (
            x_kv_69
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_13 = multi_head_attention_forward_13[0]
        multi_head_attention_forward_13 = None
        out_13 = attn_output_13.transpose(0, 1)
        attn_output_13 = None
        dropout_42 = torch.nn.functional.dropout(out_13, 0.0, False, False)
        out_13 = None
        add_42 = 0.0 + dropout_42
        dropout_42 = None
        x_46 = x_45 + add_42
        x_45 = add_42 = None
        layer_norm_44 = torch.nn.functional.layer_norm(
            x_46,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_norm2_parameters_bias_ = (None)
        input_66 = torch._C._nn.linear(
            layer_norm_44,
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_44 = l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_67 = torch._C._nn.gelu(input_66, approximate="none")
        input_66 = None
        input_68 = torch.nn.functional.dropout(input_67, 0.0, False, False)
        input_67 = None
        input_69 = torch._C._nn.linear(
            input_68,
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_68 = l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_70 = torch.nn.functional.dropout(input_69, 0.0, False, False)
        input_69 = None
        x_47 = x_46 + input_70
        x_46 = input_70 = None
        x_q_28 = torch.nn.functional.layer_norm(
            x_47,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_norm1_parameters_bias_ = (None)
        transpose_79 = x_q_28.transpose(1, 2)
        x_kv_70 = transpose_79.reshape(1, 320, 32, 32)
        transpose_79 = None
        x_kv_71 = torch.conv2d(
            x_kv_70,
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_70 = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_20 = x_kv_71.flatten(2)
        x_kv_71 = None
        transpose_80 = flatten_20.transpose(1, 2)
        flatten_20 = None
        x_kv_72 = transpose_80.contiguous()
        transpose_80 = None
        x_kv_73 = torch.nn.functional.layer_norm(
            x_kv_72,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_72 = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_29 = x_q_28.transpose(0, 1)
        x_q_28 = None
        x_kv_74 = x_kv_73.transpose(0, 1)
        x_kv_73 = None
        multi_head_attention_forward_14 = torch.nn.functional.multi_head_attention_forward(
            x_q_29,
            x_kv_74,
            x_kv_74,
            320,
            5,
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_29 = (
            x_kv_74
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_14 = multi_head_attention_forward_14[0]
        multi_head_attention_forward_14 = None
        out_14 = attn_output_14.transpose(0, 1)
        attn_output_14 = None
        dropout_45 = torch.nn.functional.dropout(out_14, 0.0, False, False)
        out_14 = None
        add_45 = 0.0 + dropout_45
        dropout_45 = None
        x_48 = x_47 + add_45
        x_47 = add_45 = None
        layer_norm_47 = torch.nn.functional.layer_norm(
            x_48,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_norm2_parameters_bias_ = (None)
        input_71 = torch._C._nn.linear(
            layer_norm_47,
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_47 = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_72 = torch._C._nn.gelu(input_71, approximate="none")
        input_71 = None
        input_73 = torch.nn.functional.dropout(input_72, 0.0, False, False)
        input_72 = None
        input_74 = torch._C._nn.linear(
            input_73,
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_73 = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_75 = torch.nn.functional.dropout(input_74, 0.0, False, False)
        input_74 = None
        x_49 = x_48 + input_75
        x_48 = input_75 = None
        x_q_30 = torch.nn.functional.layer_norm(
            x_49,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_norm1_parameters_bias_ = (None)
        transpose_84 = x_q_30.transpose(1, 2)
        x_kv_75 = transpose_84.reshape(1, 320, 32, 32)
        transpose_84 = None
        x_kv_76 = torch.conv2d(
            x_kv_75,
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_75 = l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_21 = x_kv_76.flatten(2)
        x_kv_76 = None
        transpose_85 = flatten_21.transpose(1, 2)
        flatten_21 = None
        x_kv_77 = transpose_85.contiguous()
        transpose_85 = None
        x_kv_78 = torch.nn.functional.layer_norm(
            x_kv_77,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_77 = l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_31 = x_q_30.transpose(0, 1)
        x_q_30 = None
        x_kv_79 = x_kv_78.transpose(0, 1)
        x_kv_78 = None
        multi_head_attention_forward_15 = torch.nn.functional.multi_head_attention_forward(
            x_q_31,
            x_kv_79,
            x_kv_79,
            320,
            5,
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_31 = (
            x_kv_79
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_15 = multi_head_attention_forward_15[0]
        multi_head_attention_forward_15 = None
        out_15 = attn_output_15.transpose(0, 1)
        attn_output_15 = None
        dropout_48 = torch.nn.functional.dropout(out_15, 0.0, False, False)
        out_15 = None
        add_48 = 0.0 + dropout_48
        dropout_48 = None
        x_50 = x_49 + add_48
        x_49 = add_48 = None
        layer_norm_50 = torch.nn.functional.layer_norm(
            x_50,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_norm2_parameters_bias_ = (None)
        input_76 = torch._C._nn.linear(
            layer_norm_50,
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_50 = l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_77 = torch._C._nn.gelu(input_76, approximate="none")
        input_76 = None
        input_78 = torch.nn.functional.dropout(input_77, 0.0, False, False)
        input_77 = None
        input_79 = torch._C._nn.linear(
            input_78,
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_78 = l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_80 = torch.nn.functional.dropout(input_79, 0.0, False, False)
        input_79 = None
        x_51 = x_50 + input_80
        x_50 = input_80 = None
        x_q_32 = torch.nn.functional.layer_norm(
            x_51,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_norm1_parameters_bias_ = (None)
        transpose_89 = x_q_32.transpose(1, 2)
        x_kv_80 = transpose_89.reshape(1, 320, 32, 32)
        transpose_89 = None
        x_kv_81 = torch.conv2d(
            x_kv_80,
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_80 = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_22 = x_kv_81.flatten(2)
        x_kv_81 = None
        transpose_90 = flatten_22.transpose(1, 2)
        flatten_22 = None
        x_kv_82 = transpose_90.contiguous()
        transpose_90 = None
        x_kv_83 = torch.nn.functional.layer_norm(
            x_kv_82,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_82 = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_33 = x_q_32.transpose(0, 1)
        x_q_32 = None
        x_kv_84 = x_kv_83.transpose(0, 1)
        x_kv_83 = None
        multi_head_attention_forward_16 = torch.nn.functional.multi_head_attention_forward(
            x_q_33,
            x_kv_84,
            x_kv_84,
            320,
            5,
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_33 = (
            x_kv_84
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_16 = multi_head_attention_forward_16[0]
        multi_head_attention_forward_16 = None
        out_16 = attn_output_16.transpose(0, 1)
        attn_output_16 = None
        dropout_51 = torch.nn.functional.dropout(out_16, 0.0, False, False)
        out_16 = None
        add_51 = 0.0 + dropout_51
        dropout_51 = None
        x_52 = x_51 + add_51
        x_51 = add_51 = None
        layer_norm_53 = torch.nn.functional.layer_norm(
            x_52,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_norm2_parameters_bias_ = (None)
        input_81 = torch._C._nn.linear(
            layer_norm_53,
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_53 = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_82 = torch._C._nn.gelu(input_81, approximate="none")
        input_81 = None
        input_83 = torch.nn.functional.dropout(input_82, 0.0, False, False)
        input_82 = None
        input_84 = torch._C._nn.linear(
            input_83,
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_83 = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_85 = torch.nn.functional.dropout(input_84, 0.0, False, False)
        input_84 = None
        x_53 = x_52 + input_85
        x_52 = input_85 = None
        x_q_34 = torch.nn.functional.layer_norm(
            x_53,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_norm1_parameters_bias_ = (None)
        transpose_94 = x_q_34.transpose(1, 2)
        x_kv_85 = transpose_94.reshape(1, 320, 32, 32)
        transpose_94 = None
        x_kv_86 = torch.conv2d(
            x_kv_85,
            l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_85 = l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_23 = x_kv_86.flatten(2)
        x_kv_86 = None
        transpose_95 = flatten_23.transpose(1, 2)
        flatten_23 = None
        x_kv_87 = transpose_95.contiguous()
        transpose_95 = None
        x_kv_88 = torch.nn.functional.layer_norm(
            x_kv_87,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_87 = l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_35 = x_q_34.transpose(0, 1)
        x_q_34 = None
        x_kv_89 = x_kv_88.transpose(0, 1)
        x_kv_88 = None
        multi_head_attention_forward_17 = torch.nn.functional.multi_head_attention_forward(
            x_q_35,
            x_kv_89,
            x_kv_89,
            320,
            5,
            l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_35 = (
            x_kv_89
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_17 = multi_head_attention_forward_17[0]
        multi_head_attention_forward_17 = None
        out_17 = attn_output_17.transpose(0, 1)
        attn_output_17 = None
        dropout_54 = torch.nn.functional.dropout(out_17, 0.0, False, False)
        out_17 = None
        add_54 = 0.0 + dropout_54
        dropout_54 = None
        x_54 = x_53 + add_54
        x_53 = add_54 = None
        layer_norm_56 = torch.nn.functional.layer_norm(
            x_54,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_norm2_parameters_bias_ = (None)
        input_86 = torch._C._nn.linear(
            layer_norm_56,
            l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_56 = l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_87 = torch._C._nn.gelu(input_86, approximate="none")
        input_86 = None
        input_88 = torch.nn.functional.dropout(input_87, 0.0, False, False)
        input_87 = None
        input_89 = torch._C._nn.linear(
            input_88,
            l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_88 = l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_90 = torch.nn.functional.dropout(input_89, 0.0, False, False)
        input_89 = None
        x_55 = x_54 + input_90
        x_54 = input_90 = None
        x_q_36 = torch.nn.functional.layer_norm(
            x_55,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_norm1_parameters_bias_ = (None)
        transpose_99 = x_q_36.transpose(1, 2)
        x_kv_90 = transpose_99.reshape(1, 320, 32, 32)
        transpose_99 = None
        x_kv_91 = torch.conv2d(
            x_kv_90,
            l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_90 = l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_24 = x_kv_91.flatten(2)
        x_kv_91 = None
        transpose_100 = flatten_24.transpose(1, 2)
        flatten_24 = None
        x_kv_92 = transpose_100.contiguous()
        transpose_100 = None
        x_kv_93 = torch.nn.functional.layer_norm(
            x_kv_92,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_92 = l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_37 = x_q_36.transpose(0, 1)
        x_q_36 = None
        x_kv_94 = x_kv_93.transpose(0, 1)
        x_kv_93 = None
        multi_head_attention_forward_18 = torch.nn.functional.multi_head_attention_forward(
            x_q_37,
            x_kv_94,
            x_kv_94,
            320,
            5,
            l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_37 = (
            x_kv_94
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_18 = multi_head_attention_forward_18[0]
        multi_head_attention_forward_18 = None
        out_18 = attn_output_18.transpose(0, 1)
        attn_output_18 = None
        dropout_57 = torch.nn.functional.dropout(out_18, 0.0, False, False)
        out_18 = None
        add_57 = 0.0 + dropout_57
        dropout_57 = None
        x_56 = x_55 + add_57
        x_55 = add_57 = None
        layer_norm_59 = torch.nn.functional.layer_norm(
            x_56,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_norm2_parameters_bias_ = (None)
        input_91 = torch._C._nn.linear(
            layer_norm_59,
            l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_59 = l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_92 = torch._C._nn.gelu(input_91, approximate="none")
        input_91 = None
        input_93 = torch.nn.functional.dropout(input_92, 0.0, False, False)
        input_92 = None
        input_94 = torch._C._nn.linear(
            input_93,
            l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_93 = l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_95 = torch.nn.functional.dropout(input_94, 0.0, False, False)
        input_94 = None
        x_57 = x_56 + input_95
        x_56 = input_95 = None
        x_q_38 = torch.nn.functional.layer_norm(
            x_57,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_norm1_parameters_bias_ = (None)
        transpose_104 = x_q_38.transpose(1, 2)
        x_kv_95 = transpose_104.reshape(1, 320, 32, 32)
        transpose_104 = None
        x_kv_96 = torch.conv2d(
            x_kv_95,
            l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_95 = l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_25 = x_kv_96.flatten(2)
        x_kv_96 = None
        transpose_105 = flatten_25.transpose(1, 2)
        flatten_25 = None
        x_kv_97 = transpose_105.contiguous()
        transpose_105 = None
        x_kv_98 = torch.nn.functional.layer_norm(
            x_kv_97,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_97 = l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_39 = x_q_38.transpose(0, 1)
        x_q_38 = None
        x_kv_99 = x_kv_98.transpose(0, 1)
        x_kv_98 = None
        multi_head_attention_forward_19 = torch.nn.functional.multi_head_attention_forward(
            x_q_39,
            x_kv_99,
            x_kv_99,
            320,
            5,
            l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_39 = (
            x_kv_99
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_19 = multi_head_attention_forward_19[0]
        multi_head_attention_forward_19 = None
        out_19 = attn_output_19.transpose(0, 1)
        attn_output_19 = None
        dropout_60 = torch.nn.functional.dropout(out_19, 0.0, False, False)
        out_19 = None
        add_60 = 0.0 + dropout_60
        dropout_60 = None
        x_58 = x_57 + add_60
        x_57 = add_60 = None
        layer_norm_62 = torch.nn.functional.layer_norm(
            x_58,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_norm2_parameters_bias_ = (None)
        input_96 = torch._C._nn.linear(
            layer_norm_62,
            l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_62 = l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_97 = torch._C._nn.gelu(input_96, approximate="none")
        input_96 = None
        input_98 = torch.nn.functional.dropout(input_97, 0.0, False, False)
        input_97 = None
        input_99 = torch._C._nn.linear(
            input_98,
            l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_98 = l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_100 = torch.nn.functional.dropout(input_99, 0.0, False, False)
        input_99 = None
        x_59 = x_58 + input_100
        x_58 = input_100 = None
        x_q_40 = torch.nn.functional.layer_norm(
            x_59,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_norm1_parameters_bias_ = (None)
        transpose_109 = x_q_40.transpose(1, 2)
        x_kv_100 = transpose_109.reshape(1, 320, 32, 32)
        transpose_109 = None
        x_kv_101 = torch.conv2d(
            x_kv_100,
            l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_100 = l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_26 = x_kv_101.flatten(2)
        x_kv_101 = None
        transpose_110 = flatten_26.transpose(1, 2)
        flatten_26 = None
        x_kv_102 = transpose_110.contiguous()
        transpose_110 = None
        x_kv_103 = torch.nn.functional.layer_norm(
            x_kv_102,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_102 = l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_41 = x_q_40.transpose(0, 1)
        x_q_40 = None
        x_kv_104 = x_kv_103.transpose(0, 1)
        x_kv_103 = None
        multi_head_attention_forward_20 = torch.nn.functional.multi_head_attention_forward(
            x_q_41,
            x_kv_104,
            x_kv_104,
            320,
            5,
            l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_41 = (
            x_kv_104
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_20 = multi_head_attention_forward_20[0]
        multi_head_attention_forward_20 = None
        out_20 = attn_output_20.transpose(0, 1)
        attn_output_20 = None
        dropout_63 = torch.nn.functional.dropout(out_20, 0.0, False, False)
        out_20 = None
        add_63 = 0.0 + dropout_63
        dropout_63 = None
        x_60 = x_59 + add_63
        x_59 = add_63 = None
        layer_norm_65 = torch.nn.functional.layer_norm(
            x_60,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_norm2_parameters_bias_ = (None)
        input_101 = torch._C._nn.linear(
            layer_norm_65,
            l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_65 = l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_102 = torch._C._nn.gelu(input_101, approximate="none")
        input_101 = None
        input_103 = torch.nn.functional.dropout(input_102, 0.0, False, False)
        input_102 = None
        input_104 = torch._C._nn.linear(
            input_103,
            l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_103 = l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_105 = torch.nn.functional.dropout(input_104, 0.0, False, False)
        input_104 = None
        x_61 = x_60 + input_105
        x_60 = input_105 = None
        x_q_42 = torch.nn.functional.layer_norm(
            x_61,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_norm1_parameters_bias_ = (None)
        transpose_114 = x_q_42.transpose(1, 2)
        x_kv_105 = transpose_114.reshape(1, 320, 32, 32)
        transpose_114 = None
        x_kv_106 = torch.conv2d(
            x_kv_105,
            l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_105 = l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_27 = x_kv_106.flatten(2)
        x_kv_106 = None
        transpose_115 = flatten_27.transpose(1, 2)
        flatten_27 = None
        x_kv_107 = transpose_115.contiguous()
        transpose_115 = None
        x_kv_108 = torch.nn.functional.layer_norm(
            x_kv_107,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_107 = l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_43 = x_q_42.transpose(0, 1)
        x_q_42 = None
        x_kv_109 = x_kv_108.transpose(0, 1)
        x_kv_108 = None
        multi_head_attention_forward_21 = torch.nn.functional.multi_head_attention_forward(
            x_q_43,
            x_kv_109,
            x_kv_109,
            320,
            5,
            l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_43 = (
            x_kv_109
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_21 = multi_head_attention_forward_21[0]
        multi_head_attention_forward_21 = None
        out_21 = attn_output_21.transpose(0, 1)
        attn_output_21 = None
        dropout_66 = torch.nn.functional.dropout(out_21, 0.0, False, False)
        out_21 = None
        add_66 = 0.0 + dropout_66
        dropout_66 = None
        x_62 = x_61 + add_66
        x_61 = add_66 = None
        layer_norm_68 = torch.nn.functional.layer_norm(
            x_62,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_norm2_parameters_bias_ = (None)
        input_106 = torch._C._nn.linear(
            layer_norm_68,
            l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_68 = l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_107 = torch._C._nn.gelu(input_106, approximate="none")
        input_106 = None
        input_108 = torch.nn.functional.dropout(input_107, 0.0, False, False)
        input_107 = None
        input_109 = torch._C._nn.linear(
            input_108,
            l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_108 = l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_110 = torch.nn.functional.dropout(input_109, 0.0, False, False)
        input_109 = None
        x_63 = x_62 + input_110
        x_62 = input_110 = None
        x_q_44 = torch.nn.functional.layer_norm(
            x_63,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_norm1_parameters_bias_ = (None)
        transpose_119 = x_q_44.transpose(1, 2)
        x_kv_110 = transpose_119.reshape(1, 320, 32, 32)
        transpose_119 = None
        x_kv_111 = torch.conv2d(
            x_kv_110,
            l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_110 = l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_28 = x_kv_111.flatten(2)
        x_kv_111 = None
        transpose_120 = flatten_28.transpose(1, 2)
        flatten_28 = None
        x_kv_112 = transpose_120.contiguous()
        transpose_120 = None
        x_kv_113 = torch.nn.functional.layer_norm(
            x_kv_112,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_112 = l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_45 = x_q_44.transpose(0, 1)
        x_q_44 = None
        x_kv_114 = x_kv_113.transpose(0, 1)
        x_kv_113 = None
        multi_head_attention_forward_22 = torch.nn.functional.multi_head_attention_forward(
            x_q_45,
            x_kv_114,
            x_kv_114,
            320,
            5,
            l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_45 = (
            x_kv_114
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_22 = multi_head_attention_forward_22[0]
        multi_head_attention_forward_22 = None
        out_22 = attn_output_22.transpose(0, 1)
        attn_output_22 = None
        dropout_69 = torch.nn.functional.dropout(out_22, 0.0, False, False)
        out_22 = None
        add_69 = 0.0 + dropout_69
        dropout_69 = None
        x_64 = x_63 + add_69
        x_63 = add_69 = None
        layer_norm_71 = torch.nn.functional.layer_norm(
            x_64,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_norm2_parameters_bias_ = (None)
        input_111 = torch._C._nn.linear(
            layer_norm_71,
            l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_71 = l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_112 = torch._C._nn.gelu(input_111, approximate="none")
        input_111 = None
        input_113 = torch.nn.functional.dropout(input_112, 0.0, False, False)
        input_112 = None
        input_114 = torch._C._nn.linear(
            input_113,
            l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_113 = l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_115 = torch.nn.functional.dropout(input_114, 0.0, False, False)
        input_114 = None
        x_65 = x_64 + input_115
        x_64 = input_115 = None
        x_q_46 = torch.nn.functional.layer_norm(
            x_65,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_norm1_parameters_bias_ = (None)
        transpose_124 = x_q_46.transpose(1, 2)
        x_kv_115 = transpose_124.reshape(1, 320, 32, 32)
        transpose_124 = None
        x_kv_116 = torch.conv2d(
            x_kv_115,
            l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_115 = l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_29 = x_kv_116.flatten(2)
        x_kv_116 = None
        transpose_125 = flatten_29.transpose(1, 2)
        flatten_29 = None
        x_kv_117 = transpose_125.contiguous()
        transpose_125 = None
        x_kv_118 = torch.nn.functional.layer_norm(
            x_kv_117,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_117 = l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_47 = x_q_46.transpose(0, 1)
        x_q_46 = None
        x_kv_119 = x_kv_118.transpose(0, 1)
        x_kv_118 = None
        multi_head_attention_forward_23 = torch.nn.functional.multi_head_attention_forward(
            x_q_47,
            x_kv_119,
            x_kv_119,
            320,
            5,
            l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_47 = (
            x_kv_119
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_23 = multi_head_attention_forward_23[0]
        multi_head_attention_forward_23 = None
        out_23 = attn_output_23.transpose(0, 1)
        attn_output_23 = None
        dropout_72 = torch.nn.functional.dropout(out_23, 0.0, False, False)
        out_23 = None
        add_72 = 0.0 + dropout_72
        dropout_72 = None
        x_66 = x_65 + add_72
        x_65 = add_72 = None
        layer_norm_74 = torch.nn.functional.layer_norm(
            x_66,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_norm2_parameters_bias_ = (None)
        input_116 = torch._C._nn.linear(
            layer_norm_74,
            l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_74 = l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_117 = torch._C._nn.gelu(input_116, approximate="none")
        input_116 = None
        input_118 = torch.nn.functional.dropout(input_117, 0.0, False, False)
        input_117 = None
        input_119 = torch._C._nn.linear(
            input_118,
            l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_118 = l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_120 = torch.nn.functional.dropout(input_119, 0.0, False, False)
        input_119 = None
        x_67 = x_66 + input_120
        x_66 = input_120 = None
        x_q_48 = torch.nn.functional.layer_norm(
            x_67,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_norm1_parameters_bias_ = (None)
        transpose_129 = x_q_48.transpose(1, 2)
        x_kv_120 = transpose_129.reshape(1, 320, 32, 32)
        transpose_129 = None
        x_kv_121 = torch.conv2d(
            x_kv_120,
            l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_120 = l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_30 = x_kv_121.flatten(2)
        x_kv_121 = None
        transpose_130 = flatten_30.transpose(1, 2)
        flatten_30 = None
        x_kv_122 = transpose_130.contiguous()
        transpose_130 = None
        x_kv_123 = torch.nn.functional.layer_norm(
            x_kv_122,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_122 = l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_49 = x_q_48.transpose(0, 1)
        x_q_48 = None
        x_kv_124 = x_kv_123.transpose(0, 1)
        x_kv_123 = None
        multi_head_attention_forward_24 = torch.nn.functional.multi_head_attention_forward(
            x_q_49,
            x_kv_124,
            x_kv_124,
            320,
            5,
            l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_49 = (
            x_kv_124
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_24 = multi_head_attention_forward_24[0]
        multi_head_attention_forward_24 = None
        out_24 = attn_output_24.transpose(0, 1)
        attn_output_24 = None
        dropout_75 = torch.nn.functional.dropout(out_24, 0.0, False, False)
        out_24 = None
        add_75 = 0.0 + dropout_75
        dropout_75 = None
        x_68 = x_67 + add_75
        x_67 = add_75 = None
        layer_norm_77 = torch.nn.functional.layer_norm(
            x_68,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_norm2_parameters_bias_ = (None)
        input_121 = torch._C._nn.linear(
            layer_norm_77,
            l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_77 = l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_122 = torch._C._nn.gelu(input_121, approximate="none")
        input_121 = None
        input_123 = torch.nn.functional.dropout(input_122, 0.0, False, False)
        input_122 = None
        input_124 = torch._C._nn.linear(
            input_123,
            l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_123 = l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_125 = torch.nn.functional.dropout(input_124, 0.0, False, False)
        input_124 = None
        x_69 = x_68 + input_125
        x_68 = input_125 = None
        x_q_50 = torch.nn.functional.layer_norm(
            x_69,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_norm1_parameters_bias_ = (None)
        transpose_134 = x_q_50.transpose(1, 2)
        x_kv_125 = transpose_134.reshape(1, 320, 32, 32)
        transpose_134 = None
        x_kv_126 = torch.conv2d(
            x_kv_125,
            l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_125 = l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_31 = x_kv_126.flatten(2)
        x_kv_126 = None
        transpose_135 = flatten_31.transpose(1, 2)
        flatten_31 = None
        x_kv_127 = transpose_135.contiguous()
        transpose_135 = None
        x_kv_128 = torch.nn.functional.layer_norm(
            x_kv_127,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_127 = l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_51 = x_q_50.transpose(0, 1)
        x_q_50 = None
        x_kv_129 = x_kv_128.transpose(0, 1)
        x_kv_128 = None
        multi_head_attention_forward_25 = torch.nn.functional.multi_head_attention_forward(
            x_q_51,
            x_kv_129,
            x_kv_129,
            320,
            5,
            l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_51 = (
            x_kv_129
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_25 = multi_head_attention_forward_25[0]
        multi_head_attention_forward_25 = None
        out_25 = attn_output_25.transpose(0, 1)
        attn_output_25 = None
        dropout_78 = torch.nn.functional.dropout(out_25, 0.0, False, False)
        out_25 = None
        add_78 = 0.0 + dropout_78
        dropout_78 = None
        x_70 = x_69 + add_78
        x_69 = add_78 = None
        layer_norm_80 = torch.nn.functional.layer_norm(
            x_70,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_norm2_parameters_bias_ = (None)
        input_126 = torch._C._nn.linear(
            layer_norm_80,
            l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_80 = l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_127 = torch._C._nn.gelu(input_126, approximate="none")
        input_126 = None
        input_128 = torch.nn.functional.dropout(input_127, 0.0, False, False)
        input_127 = None
        input_129 = torch._C._nn.linear(
            input_128,
            l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_128 = l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_130 = torch.nn.functional.dropout(input_129, 0.0, False, False)
        input_129 = None
        x_71 = x_70 + input_130
        x_70 = input_130 = None
        x_q_52 = torch.nn.functional.layer_norm(
            x_71,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_norm1_parameters_bias_ = (None)
        transpose_139 = x_q_52.transpose(1, 2)
        x_kv_130 = transpose_139.reshape(1, 320, 32, 32)
        transpose_139 = None
        x_kv_131 = torch.conv2d(
            x_kv_130,
            l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_130 = l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_32 = x_kv_131.flatten(2)
        x_kv_131 = None
        transpose_140 = flatten_32.transpose(1, 2)
        flatten_32 = None
        x_kv_132 = transpose_140.contiguous()
        transpose_140 = None
        x_kv_133 = torch.nn.functional.layer_norm(
            x_kv_132,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_132 = l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_53 = x_q_52.transpose(0, 1)
        x_q_52 = None
        x_kv_134 = x_kv_133.transpose(0, 1)
        x_kv_133 = None
        multi_head_attention_forward_26 = torch.nn.functional.multi_head_attention_forward(
            x_q_53,
            x_kv_134,
            x_kv_134,
            320,
            5,
            l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_53 = (
            x_kv_134
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_26 = multi_head_attention_forward_26[0]
        multi_head_attention_forward_26 = None
        out_26 = attn_output_26.transpose(0, 1)
        attn_output_26 = None
        dropout_81 = torch.nn.functional.dropout(out_26, 0.0, False, False)
        out_26 = None
        add_81 = 0.0 + dropout_81
        dropout_81 = None
        x_72 = x_71 + add_81
        x_71 = add_81 = None
        layer_norm_83 = torch.nn.functional.layer_norm(
            x_72,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_norm2_parameters_bias_ = (None)
        input_131 = torch._C._nn.linear(
            layer_norm_83,
            l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_83 = l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_132 = torch._C._nn.gelu(input_131, approximate="none")
        input_131 = None
        input_133 = torch.nn.functional.dropout(input_132, 0.0, False, False)
        input_132 = None
        input_134 = torch._C._nn.linear(
            input_133,
            l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_133 = l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_135 = torch.nn.functional.dropout(input_134, 0.0, False, False)
        input_134 = None
        x_73 = x_72 + input_135
        x_72 = input_135 = None
        x_q_54 = torch.nn.functional.layer_norm(
            x_73,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_norm1_parameters_bias_ = (None)
        transpose_144 = x_q_54.transpose(1, 2)
        x_kv_135 = transpose_144.reshape(1, 320, 32, 32)
        transpose_144 = None
        x_kv_136 = torch.conv2d(
            x_kv_135,
            l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_135 = l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_33 = x_kv_136.flatten(2)
        x_kv_136 = None
        transpose_145 = flatten_33.transpose(1, 2)
        flatten_33 = None
        x_kv_137 = transpose_145.contiguous()
        transpose_145 = None
        x_kv_138 = torch.nn.functional.layer_norm(
            x_kv_137,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_137 = l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_55 = x_q_54.transpose(0, 1)
        x_q_54 = None
        x_kv_139 = x_kv_138.transpose(0, 1)
        x_kv_138 = None
        multi_head_attention_forward_27 = torch.nn.functional.multi_head_attention_forward(
            x_q_55,
            x_kv_139,
            x_kv_139,
            320,
            5,
            l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_55 = (
            x_kv_139
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_27 = multi_head_attention_forward_27[0]
        multi_head_attention_forward_27 = None
        out_27 = attn_output_27.transpose(0, 1)
        attn_output_27 = None
        dropout_84 = torch.nn.functional.dropout(out_27, 0.0, False, False)
        out_27 = None
        add_84 = 0.0 + dropout_84
        dropout_84 = None
        x_74 = x_73 + add_84
        x_73 = add_84 = None
        layer_norm_86 = torch.nn.functional.layer_norm(
            x_74,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_norm2_parameters_bias_ = (None)
        input_136 = torch._C._nn.linear(
            layer_norm_86,
            l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_86 = l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_137 = torch._C._nn.gelu(input_136, approximate="none")
        input_136 = None
        input_138 = torch.nn.functional.dropout(input_137, 0.0, False, False)
        input_137 = None
        input_139 = torch._C._nn.linear(
            input_138,
            l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_138 = l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_140 = torch.nn.functional.dropout(input_139, 0.0, False, False)
        input_139 = None
        x_75 = x_74 + input_140
        x_74 = input_140 = None
        x_q_56 = torch.nn.functional.layer_norm(
            x_75,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_norm1_parameters_bias_ = (None)
        transpose_149 = x_q_56.transpose(1, 2)
        x_kv_140 = transpose_149.reshape(1, 320, 32, 32)
        transpose_149 = None
        x_kv_141 = torch.conv2d(
            x_kv_140,
            l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_140 = l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_34 = x_kv_141.flatten(2)
        x_kv_141 = None
        transpose_150 = flatten_34.transpose(1, 2)
        flatten_34 = None
        x_kv_142 = transpose_150.contiguous()
        transpose_150 = None
        x_kv_143 = torch.nn.functional.layer_norm(
            x_kv_142,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_142 = l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_57 = x_q_56.transpose(0, 1)
        x_q_56 = None
        x_kv_144 = x_kv_143.transpose(0, 1)
        x_kv_143 = None
        multi_head_attention_forward_28 = torch.nn.functional.multi_head_attention_forward(
            x_q_57,
            x_kv_144,
            x_kv_144,
            320,
            5,
            l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_57 = (
            x_kv_144
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_28 = multi_head_attention_forward_28[0]
        multi_head_attention_forward_28 = None
        out_28 = attn_output_28.transpose(0, 1)
        attn_output_28 = None
        dropout_87 = torch.nn.functional.dropout(out_28, 0.0, False, False)
        out_28 = None
        add_87 = 0.0 + dropout_87
        dropout_87 = None
        x_76 = x_75 + add_87
        x_75 = add_87 = None
        layer_norm_89 = torch.nn.functional.layer_norm(
            x_76,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_norm2_parameters_bias_ = (None)
        input_141 = torch._C._nn.linear(
            layer_norm_89,
            l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_89 = l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_142 = torch._C._nn.gelu(input_141, approximate="none")
        input_141 = None
        input_143 = torch.nn.functional.dropout(input_142, 0.0, False, False)
        input_142 = None
        input_144 = torch._C._nn.linear(
            input_143,
            l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_143 = l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_145 = torch.nn.functional.dropout(input_144, 0.0, False, False)
        input_144 = None
        x_77 = x_76 + input_145
        x_76 = input_145 = None
        x_q_58 = torch.nn.functional.layer_norm(
            x_77,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_norm1_parameters_bias_ = (None)
        transpose_154 = x_q_58.transpose(1, 2)
        x_kv_145 = transpose_154.reshape(1, 320, 32, 32)
        transpose_154 = None
        x_kv_146 = torch.conv2d(
            x_kv_145,
            l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_145 = l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_35 = x_kv_146.flatten(2)
        x_kv_146 = None
        transpose_155 = flatten_35.transpose(1, 2)
        flatten_35 = None
        x_kv_147 = transpose_155.contiguous()
        transpose_155 = None
        x_kv_148 = torch.nn.functional.layer_norm(
            x_kv_147,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_147 = l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_59 = x_q_58.transpose(0, 1)
        x_q_58 = None
        x_kv_149 = x_kv_148.transpose(0, 1)
        x_kv_148 = None
        multi_head_attention_forward_29 = torch.nn.functional.multi_head_attention_forward(
            x_q_59,
            x_kv_149,
            x_kv_149,
            320,
            5,
            l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_59 = (
            x_kv_149
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_29 = multi_head_attention_forward_29[0]
        multi_head_attention_forward_29 = None
        out_29 = attn_output_29.transpose(0, 1)
        attn_output_29 = None
        dropout_90 = torch.nn.functional.dropout(out_29, 0.0, False, False)
        out_29 = None
        add_90 = 0.0 + dropout_90
        dropout_90 = None
        x_78 = x_77 + add_90
        x_77 = add_90 = None
        layer_norm_92 = torch.nn.functional.layer_norm(
            x_78,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_norm2_parameters_bias_ = (None)
        input_146 = torch._C._nn.linear(
            layer_norm_92,
            l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_92 = l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_147 = torch._C._nn.gelu(input_146, approximate="none")
        input_146 = None
        input_148 = torch.nn.functional.dropout(input_147, 0.0, False, False)
        input_147 = None
        input_149 = torch._C._nn.linear(
            input_148,
            l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_148 = l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_18_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_150 = torch.nn.functional.dropout(input_149, 0.0, False, False)
        input_149 = None
        x_79 = x_78 + input_150
        x_78 = input_150 = None
        x_q_60 = torch.nn.functional.layer_norm(
            x_79,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_norm1_parameters_bias_ = (None)
        transpose_159 = x_q_60.transpose(1, 2)
        x_kv_150 = transpose_159.reshape(1, 320, 32, 32)
        transpose_159 = None
        x_kv_151 = torch.conv2d(
            x_kv_150,
            l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_150 = l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_36 = x_kv_151.flatten(2)
        x_kv_151 = None
        transpose_160 = flatten_36.transpose(1, 2)
        flatten_36 = None
        x_kv_152 = transpose_160.contiguous()
        transpose_160 = None
        x_kv_153 = torch.nn.functional.layer_norm(
            x_kv_152,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_152 = l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_61 = x_q_60.transpose(0, 1)
        x_q_60 = None
        x_kv_154 = x_kv_153.transpose(0, 1)
        x_kv_153 = None
        multi_head_attention_forward_30 = torch.nn.functional.multi_head_attention_forward(
            x_q_61,
            x_kv_154,
            x_kv_154,
            320,
            5,
            l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_61 = (
            x_kv_154
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_30 = multi_head_attention_forward_30[0]
        multi_head_attention_forward_30 = None
        out_30 = attn_output_30.transpose(0, 1)
        attn_output_30 = None
        dropout_93 = torch.nn.functional.dropout(out_30, 0.0, False, False)
        out_30 = None
        add_93 = 0.0 + dropout_93
        dropout_93 = None
        x_80 = x_79 + add_93
        x_79 = add_93 = None
        layer_norm_95 = torch.nn.functional.layer_norm(
            x_80,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_norm2_parameters_bias_ = (None)
        input_151 = torch._C._nn.linear(
            layer_norm_95,
            l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_95 = l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_152 = torch._C._nn.gelu(input_151, approximate="none")
        input_151 = None
        input_153 = torch.nn.functional.dropout(input_152, 0.0, False, False)
        input_152 = None
        input_154 = torch._C._nn.linear(
            input_153,
            l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_153 = l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_19_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_155 = torch.nn.functional.dropout(input_154, 0.0, False, False)
        input_154 = None
        x_81 = x_80 + input_155
        x_80 = input_155 = None
        x_q_62 = torch.nn.functional.layer_norm(
            x_81,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_norm1_parameters_bias_ = (None)
        transpose_164 = x_q_62.transpose(1, 2)
        x_kv_155 = transpose_164.reshape(1, 320, 32, 32)
        transpose_164 = None
        x_kv_156 = torch.conv2d(
            x_kv_155,
            l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_155 = l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_37 = x_kv_156.flatten(2)
        x_kv_156 = None
        transpose_165 = flatten_37.transpose(1, 2)
        flatten_37 = None
        x_kv_157 = transpose_165.contiguous()
        transpose_165 = None
        x_kv_158 = torch.nn.functional.layer_norm(
            x_kv_157,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_157 = l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_63 = x_q_62.transpose(0, 1)
        x_q_62 = None
        x_kv_159 = x_kv_158.transpose(0, 1)
        x_kv_158 = None
        multi_head_attention_forward_31 = torch.nn.functional.multi_head_attention_forward(
            x_q_63,
            x_kv_159,
            x_kv_159,
            320,
            5,
            l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_63 = (
            x_kv_159
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_31 = multi_head_attention_forward_31[0]
        multi_head_attention_forward_31 = None
        out_31 = attn_output_31.transpose(0, 1)
        attn_output_31 = None
        dropout_96 = torch.nn.functional.dropout(out_31, 0.0, False, False)
        out_31 = None
        add_96 = 0.0 + dropout_96
        dropout_96 = None
        x_82 = x_81 + add_96
        x_81 = add_96 = None
        layer_norm_98 = torch.nn.functional.layer_norm(
            x_82,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_norm2_parameters_bias_ = (None)
        input_156 = torch._C._nn.linear(
            layer_norm_98,
            l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_98 = l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_157 = torch._C._nn.gelu(input_156, approximate="none")
        input_156 = None
        input_158 = torch.nn.functional.dropout(input_157, 0.0, False, False)
        input_157 = None
        input_159 = torch._C._nn.linear(
            input_158,
            l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_158 = l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_20_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_160 = torch.nn.functional.dropout(input_159, 0.0, False, False)
        input_159 = None
        x_83 = x_82 + input_160
        x_82 = input_160 = None
        x_q_64 = torch.nn.functional.layer_norm(
            x_83,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_norm1_parameters_bias_ = (None)
        transpose_169 = x_q_64.transpose(1, 2)
        x_kv_160 = transpose_169.reshape(1, 320, 32, 32)
        transpose_169 = None
        x_kv_161 = torch.conv2d(
            x_kv_160,
            l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_160 = l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_38 = x_kv_161.flatten(2)
        x_kv_161 = None
        transpose_170 = flatten_38.transpose(1, 2)
        flatten_38 = None
        x_kv_162 = transpose_170.contiguous()
        transpose_170 = None
        x_kv_163 = torch.nn.functional.layer_norm(
            x_kv_162,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_162 = l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_65 = x_q_64.transpose(0, 1)
        x_q_64 = None
        x_kv_164 = x_kv_163.transpose(0, 1)
        x_kv_163 = None
        multi_head_attention_forward_32 = torch.nn.functional.multi_head_attention_forward(
            x_q_65,
            x_kv_164,
            x_kv_164,
            320,
            5,
            l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_65 = (
            x_kv_164
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_32 = multi_head_attention_forward_32[0]
        multi_head_attention_forward_32 = None
        out_32 = attn_output_32.transpose(0, 1)
        attn_output_32 = None
        dropout_99 = torch.nn.functional.dropout(out_32, 0.0, False, False)
        out_32 = None
        add_99 = 0.0 + dropout_99
        dropout_99 = None
        x_84 = x_83 + add_99
        x_83 = add_99 = None
        layer_norm_101 = torch.nn.functional.layer_norm(
            x_84,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_norm2_parameters_bias_ = (None)
        input_161 = torch._C._nn.linear(
            layer_norm_101,
            l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_101 = l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_162 = torch._C._nn.gelu(input_161, approximate="none")
        input_161 = None
        input_163 = torch.nn.functional.dropout(input_162, 0.0, False, False)
        input_162 = None
        input_164 = torch._C._nn.linear(
            input_163,
            l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_163 = l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_21_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_165 = torch.nn.functional.dropout(input_164, 0.0, False, False)
        input_164 = None
        x_85 = x_84 + input_165
        x_84 = input_165 = None
        x_q_66 = torch.nn.functional.layer_norm(
            x_85,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_norm1_parameters_bias_ = (None)
        transpose_174 = x_q_66.transpose(1, 2)
        x_kv_165 = transpose_174.reshape(1, 320, 32, 32)
        transpose_174 = None
        x_kv_166 = torch.conv2d(
            x_kv_165,
            l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_165 = l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_39 = x_kv_166.flatten(2)
        x_kv_166 = None
        transpose_175 = flatten_39.transpose(1, 2)
        flatten_39 = None
        x_kv_167 = transpose_175.contiguous()
        transpose_175 = None
        x_kv_168 = torch.nn.functional.layer_norm(
            x_kv_167,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_167 = l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_67 = x_q_66.transpose(0, 1)
        x_q_66 = None
        x_kv_169 = x_kv_168.transpose(0, 1)
        x_kv_168 = None
        multi_head_attention_forward_33 = torch.nn.functional.multi_head_attention_forward(
            x_q_67,
            x_kv_169,
            x_kv_169,
            320,
            5,
            l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_67 = (
            x_kv_169
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_33 = multi_head_attention_forward_33[0]
        multi_head_attention_forward_33 = None
        out_33 = attn_output_33.transpose(0, 1)
        attn_output_33 = None
        dropout_102 = torch.nn.functional.dropout(out_33, 0.0, False, False)
        out_33 = None
        add_102 = 0.0 + dropout_102
        dropout_102 = None
        x_86 = x_85 + add_102
        x_85 = add_102 = None
        layer_norm_104 = torch.nn.functional.layer_norm(
            x_86,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_norm2_parameters_bias_ = (None)
        input_166 = torch._C._nn.linear(
            layer_norm_104,
            l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_104 = l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_167 = torch._C._nn.gelu(input_166, approximate="none")
        input_166 = None
        input_168 = torch.nn.functional.dropout(input_167, 0.0, False, False)
        input_167 = None
        input_169 = torch._C._nn.linear(
            input_168,
            l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_168 = l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_22_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_170 = torch.nn.functional.dropout(input_169, 0.0, False, False)
        input_169 = None
        x_87 = x_86 + input_170
        x_86 = input_170 = None
        x_q_68 = torch.nn.functional.layer_norm(
            x_87,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_norm1_parameters_bias_ = (None)
        transpose_179 = x_q_68.transpose(1, 2)
        x_kv_170 = transpose_179.reshape(1, 320, 32, 32)
        transpose_179 = None
        x_kv_171 = torch.conv2d(
            x_kv_170,
            l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_170 = l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_40 = x_kv_171.flatten(2)
        x_kv_171 = None
        transpose_180 = flatten_40.transpose(1, 2)
        flatten_40 = None
        x_kv_172 = transpose_180.contiguous()
        transpose_180 = None
        x_kv_173 = torch.nn.functional.layer_norm(
            x_kv_172,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_172 = l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_69 = x_q_68.transpose(0, 1)
        x_q_68 = None
        x_kv_174 = x_kv_173.transpose(0, 1)
        x_kv_173 = None
        multi_head_attention_forward_34 = torch.nn.functional.multi_head_attention_forward(
            x_q_69,
            x_kv_174,
            x_kv_174,
            320,
            5,
            l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_69 = (
            x_kv_174
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_34 = multi_head_attention_forward_34[0]
        multi_head_attention_forward_34 = None
        out_34 = attn_output_34.transpose(0, 1)
        attn_output_34 = None
        dropout_105 = torch.nn.functional.dropout(out_34, 0.0, False, False)
        out_34 = None
        add_105 = 0.0 + dropout_105
        dropout_105 = None
        x_88 = x_87 + add_105
        x_87 = add_105 = None
        layer_norm_107 = torch.nn.functional.layer_norm(
            x_88,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_norm2_parameters_bias_ = (None)
        input_171 = torch._C._nn.linear(
            layer_norm_107,
            l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_107 = l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_172 = torch._C._nn.gelu(input_171, approximate="none")
        input_171 = None
        input_173 = torch.nn.functional.dropout(input_172, 0.0, False, False)
        input_172 = None
        input_174 = torch._C._nn.linear(
            input_173,
            l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_173 = l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_23_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_175 = torch.nn.functional.dropout(input_174, 0.0, False, False)
        input_174 = None
        x_89 = x_88 + input_175
        x_88 = input_175 = None
        x_q_70 = torch.nn.functional.layer_norm(
            x_89,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_norm1_parameters_bias_ = (None)
        transpose_184 = x_q_70.transpose(1, 2)
        x_kv_175 = transpose_184.reshape(1, 320, 32, 32)
        transpose_184 = None
        x_kv_176 = torch.conv2d(
            x_kv_175,
            l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_175 = l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_41 = x_kv_176.flatten(2)
        x_kv_176 = None
        transpose_185 = flatten_41.transpose(1, 2)
        flatten_41 = None
        x_kv_177 = transpose_185.contiguous()
        transpose_185 = None
        x_kv_178 = torch.nn.functional.layer_norm(
            x_kv_177,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_177 = l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_71 = x_q_70.transpose(0, 1)
        x_q_70 = None
        x_kv_179 = x_kv_178.transpose(0, 1)
        x_kv_178 = None
        multi_head_attention_forward_35 = torch.nn.functional.multi_head_attention_forward(
            x_q_71,
            x_kv_179,
            x_kv_179,
            320,
            5,
            l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_71 = (
            x_kv_179
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_35 = multi_head_attention_forward_35[0]
        multi_head_attention_forward_35 = None
        out_35 = attn_output_35.transpose(0, 1)
        attn_output_35 = None
        dropout_108 = torch.nn.functional.dropout(out_35, 0.0, False, False)
        out_35 = None
        add_108 = 0.0 + dropout_108
        dropout_108 = None
        x_90 = x_89 + add_108
        x_89 = add_108 = None
        layer_norm_110 = torch.nn.functional.layer_norm(
            x_90,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_norm2_parameters_bias_ = (None)
        input_176 = torch._C._nn.linear(
            layer_norm_110,
            l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_110 = l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_177 = torch._C._nn.gelu(input_176, approximate="none")
        input_176 = None
        input_178 = torch.nn.functional.dropout(input_177, 0.0, False, False)
        input_177 = None
        input_179 = torch._C._nn.linear(
            input_178,
            l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_178 = l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_24_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_180 = torch.nn.functional.dropout(input_179, 0.0, False, False)
        input_179 = None
        x_91 = x_90 + input_180
        x_90 = input_180 = None
        x_q_72 = torch.nn.functional.layer_norm(
            x_91,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_norm1_parameters_bias_ = (None)
        transpose_189 = x_q_72.transpose(1, 2)
        x_kv_180 = transpose_189.reshape(1, 320, 32, 32)
        transpose_189 = None
        x_kv_181 = torch.conv2d(
            x_kv_180,
            l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_180 = l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_42 = x_kv_181.flatten(2)
        x_kv_181 = None
        transpose_190 = flatten_42.transpose(1, 2)
        flatten_42 = None
        x_kv_182 = transpose_190.contiguous()
        transpose_190 = None
        x_kv_183 = torch.nn.functional.layer_norm(
            x_kv_182,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_182 = l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_73 = x_q_72.transpose(0, 1)
        x_q_72 = None
        x_kv_184 = x_kv_183.transpose(0, 1)
        x_kv_183 = None
        multi_head_attention_forward_36 = torch.nn.functional.multi_head_attention_forward(
            x_q_73,
            x_kv_184,
            x_kv_184,
            320,
            5,
            l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_73 = (
            x_kv_184
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_36 = multi_head_attention_forward_36[0]
        multi_head_attention_forward_36 = None
        out_36 = attn_output_36.transpose(0, 1)
        attn_output_36 = None
        dropout_111 = torch.nn.functional.dropout(out_36, 0.0, False, False)
        out_36 = None
        add_111 = 0.0 + dropout_111
        dropout_111 = None
        x_92 = x_91 + add_111
        x_91 = add_111 = None
        layer_norm_113 = torch.nn.functional.layer_norm(
            x_92,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_norm2_parameters_bias_ = (None)
        input_181 = torch._C._nn.linear(
            layer_norm_113,
            l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_113 = l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_182 = torch._C._nn.gelu(input_181, approximate="none")
        input_181 = None
        input_183 = torch.nn.functional.dropout(input_182, 0.0, False, False)
        input_182 = None
        input_184 = torch._C._nn.linear(
            input_183,
            l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_183 = l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_25_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_185 = torch.nn.functional.dropout(input_184, 0.0, False, False)
        input_184 = None
        x_93 = x_92 + input_185
        x_92 = input_185 = None
        x_q_74 = torch.nn.functional.layer_norm(
            x_93,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_norm1_parameters_bias_ = (None)
        transpose_194 = x_q_74.transpose(1, 2)
        x_kv_185 = transpose_194.reshape(1, 320, 32, 32)
        transpose_194 = None
        x_kv_186 = torch.conv2d(
            x_kv_185,
            l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_185 = l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_43 = x_kv_186.flatten(2)
        x_kv_186 = None
        transpose_195 = flatten_43.transpose(1, 2)
        flatten_43 = None
        x_kv_187 = transpose_195.contiguous()
        transpose_195 = None
        x_kv_188 = torch.nn.functional.layer_norm(
            x_kv_187,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_187 = l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_75 = x_q_74.transpose(0, 1)
        x_q_74 = None
        x_kv_189 = x_kv_188.transpose(0, 1)
        x_kv_188 = None
        multi_head_attention_forward_37 = torch.nn.functional.multi_head_attention_forward(
            x_q_75,
            x_kv_189,
            x_kv_189,
            320,
            5,
            l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_75 = (
            x_kv_189
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_37 = multi_head_attention_forward_37[0]
        multi_head_attention_forward_37 = None
        out_37 = attn_output_37.transpose(0, 1)
        attn_output_37 = None
        dropout_114 = torch.nn.functional.dropout(out_37, 0.0, False, False)
        out_37 = None
        add_114 = 0.0 + dropout_114
        dropout_114 = None
        x_94 = x_93 + add_114
        x_93 = add_114 = None
        layer_norm_116 = torch.nn.functional.layer_norm(
            x_94,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_norm2_parameters_bias_ = (None)
        input_186 = torch._C._nn.linear(
            layer_norm_116,
            l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_116 = l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_187 = torch._C._nn.gelu(input_186, approximate="none")
        input_186 = None
        input_188 = torch.nn.functional.dropout(input_187, 0.0, False, False)
        input_187 = None
        input_189 = torch._C._nn.linear(
            input_188,
            l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_188 = l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_26_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_190 = torch.nn.functional.dropout(input_189, 0.0, False, False)
        input_189 = None
        x_95 = x_94 + input_190
        x_94 = input_190 = None
        reshape_40 = x_95.reshape(1, 32, 32, -1)
        x_95 = None
        permute_2 = reshape_40.permute(0, 3, 1, 2)
        reshape_40 = None
        x_96 = permute_2.contiguous()
        permute_2 = None
        x_97 = torch.conv2d(
            x_96,
            l_self_modules_backbone_modules_patch_embeds_modules_3_modules_projection_parameters_weight_,
            l_self_modules_backbone_modules_patch_embeds_modules_3_modules_projection_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_patch_embeds_modules_3_modules_projection_parameters_weight_ = l_self_modules_backbone_modules_patch_embeds_modules_3_modules_projection_parameters_bias_ = (None)
        flatten_44 = x_97.flatten(2)
        x_97 = None
        x_98 = flatten_44.transpose(1, 2)
        flatten_44 = None
        x_99 = torch.nn.functional.layer_norm(
            x_98,
            (512,),
            l_self_modules_backbone_modules_patch_embeds_modules_3_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_patch_embeds_modules_3_modules_norm_parameters_bias_,
            1e-05,
        )
        x_98 = l_self_modules_backbone_modules_patch_embeds_modules_3_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_patch_embeds_modules_3_modules_norm_parameters_bias_ = (None)
        x_100 = torch.nn.functional.dropout(x_99, 0.0, False, False)
        x_99 = None
        x_q_76 = torch.nn.functional.layer_norm(
            x_100,
            (512,),
            l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm1_parameters_bias_ = (None)
        x_q_77 = x_q_76.transpose(0, 1)
        x_kv_190 = x_q_76.transpose(0, 1)
        x_q_76 = None
        multi_head_attention_forward_38 = torch.nn.functional.multi_head_attention_forward(
            x_q_77,
            x_kv_190,
            x_kv_190,
            512,
            8,
            l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_77 = (
            x_kv_190
        ) = l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_38 = multi_head_attention_forward_38[0]
        multi_head_attention_forward_38 = None
        out_38 = attn_output_38.transpose(0, 1)
        attn_output_38 = None
        dropout_118 = torch.nn.functional.dropout(out_38, 0.0, False, False)
        out_38 = None
        add_117 = 0.0 + dropout_118
        dropout_118 = None
        x_101 = x_100 + add_117
        x_100 = add_117 = None
        layer_norm_119 = torch.nn.functional.layer_norm(
            x_101,
            (512,),
            l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm2_parameters_bias_ = (None)
        input_191 = torch._C._nn.linear(
            layer_norm_119,
            l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_119 = l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_192 = torch._C._nn.gelu(input_191, approximate="none")
        input_191 = None
        input_193 = torch.nn.functional.dropout(input_192, 0.0, False, False)
        input_192 = None
        input_194 = torch._C._nn.linear(
            input_193,
            l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_193 = l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_195 = torch.nn.functional.dropout(input_194, 0.0, False, False)
        input_194 = None
        x_102 = x_101 + input_195
        x_101 = input_195 = None
        transpose_203 = x_102.transpose(1, 2)
        x_102 = None
        cnn_feat_3 = transpose_203.view(1, 512, 16, 16)
        transpose_203 = None
        conv2d_45 = torch.conv2d(
            cnn_feat_3,
            l_self_modules_backbone_modules_position_encodings_modules_3_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_position_encodings_modules_3_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        l_self_modules_backbone_modules_position_encodings_modules_3_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_position_encodings_modules_3_modules_proj_parameters_bias_ = (None)
        x_103 = conv2d_45 + cnn_feat_3
        conv2d_45 = cnn_feat_3 = None
        flatten_45 = x_103.flatten(2)
        x_103 = None
        x_104 = flatten_45.transpose(1, 2)
        flatten_45 = None
        x_q_78 = torch.nn.functional.layer_norm(
            x_104,
            (512,),
            l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_norm1_parameters_bias_ = (None)
        x_q_79 = x_q_78.transpose(0, 1)
        x_kv_191 = x_q_78.transpose(0, 1)
        x_q_78 = None
        multi_head_attention_forward_39 = torch.nn.functional.multi_head_attention_forward(
            x_q_79,
            x_kv_191,
            x_kv_191,
            512,
            8,
            l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_79 = (
            x_kv_191
        ) = l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_39 = multi_head_attention_forward_39[0]
        multi_head_attention_forward_39 = None
        out_39 = attn_output_39.transpose(0, 1)
        attn_output_39 = None
        dropout_121 = torch.nn.functional.dropout(out_39, 0.0, False, False)
        out_39 = None
        add_121 = 0.0 + dropout_121
        dropout_121 = None
        x_105 = x_104 + add_121
        x_104 = add_121 = None
        layer_norm_121 = torch.nn.functional.layer_norm(
            x_105,
            (512,),
            l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_norm2_parameters_bias_ = (None)
        input_196 = torch._C._nn.linear(
            layer_norm_121,
            l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_121 = l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_197 = torch._C._nn.gelu(input_196, approximate="none")
        input_196 = None
        input_198 = torch.nn.functional.dropout(input_197, 0.0, False, False)
        input_197 = None
        input_199 = torch._C._nn.linear(
            input_198,
            l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_198 = l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_200 = torch.nn.functional.dropout(input_199, 0.0, False, False)
        input_199 = None
        x_106 = x_105 + input_200
        x_105 = input_200 = None
        x_q_80 = torch.nn.functional.layer_norm(
            x_106,
            (512,),
            l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_norm1_parameters_bias_ = (None)
        x_q_81 = x_q_80.transpose(0, 1)
        x_kv_192 = x_q_80.transpose(0, 1)
        x_q_80 = None
        multi_head_attention_forward_40 = torch.nn.functional.multi_head_attention_forward(
            x_q_81,
            x_kv_192,
            x_kv_192,
            512,
            8,
            l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        x_q_81 = (
            x_kv_192
        ) = l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_40 = multi_head_attention_forward_40[0]
        multi_head_attention_forward_40 = None
        out_40 = attn_output_40.transpose(0, 1)
        attn_output_40 = None
        dropout_124 = torch.nn.functional.dropout(out_40, 0.0, False, False)
        out_40 = None
        add_124 = 0.0 + dropout_124
        dropout_124 = None
        x_107 = x_106 + add_124
        x_106 = add_124 = None
        layer_norm_123 = torch.nn.functional.layer_norm(
            x_107,
            (512,),
            l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_norm2_parameters_bias_ = (None)
        input_201 = torch._C._nn.linear(
            layer_norm_123,
            l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_123 = l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_202 = torch._C._nn.gelu(input_201, approximate="none")
        input_201 = None
        input_203 = torch.nn.functional.dropout(input_202, 0.0, False, False)
        input_202 = None
        input_204 = torch._C._nn.linear(
            input_203,
            l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_203 = l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_205 = torch.nn.functional.dropout(input_204, 0.0, False, False)
        input_204 = None
        x_108 = x_107 + input_205
        x_107 = input_205 = None
        reshape_41 = x_108.reshape(1, 16, 16, -1)
        x_108 = None
        permute_3 = reshape_41.permute(0, 3, 1, 2)
        reshape_41 = None
        x_109 = permute_3.contiguous()
        permute_3 = None
        x_110 = torch.conv2d(
            x_12,
            l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_12 = l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_conv_parameters_weight_ = (None)
        x_111 = torch.nn.functional.batch_norm(
            x_110,
            l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_110 = l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_bias_ = (None)
        x_112 = torch.nn.functional.relu(x_111, inplace=False)
        x_111 = None
        x_113 = torch.conv2d(
            x_35,
            l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_35 = l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_conv_parameters_weight_ = (None)
        x_114 = torch.nn.functional.batch_norm(
            x_113,
            l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_113 = l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_bias_ = (None)
        x_115 = torch.nn.functional.relu(x_114, inplace=False)
        x_114 = None
        x_116 = torch.conv2d(
            x_96,
            l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_96 = l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_conv_parameters_weight_ = (None)
        x_117 = torch.nn.functional.batch_norm(
            x_116,
            l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_116 = l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_bias_ = (None)
        x_118 = torch.nn.functional.relu(x_117, inplace=False)
        x_117 = None
        input_206 = torch.nn.functional.adaptive_avg_pool2d(x_109, 1)
        x_119 = torch.conv2d(
            input_206,
            l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_206 = l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_conv_parameters_weight_ = (None)
        x_120 = torch.nn.functional.batch_norm(
            x_119,
            l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_119 = l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_bias_ = (None)
        x_121 = torch.nn.functional.relu(x_120, inplace=True)
        x_120 = None
        upsampled_ppm_out = torch.nn.functional.interpolate(
            x_121, (16, 16), None, "bilinear", False
        )
        x_121 = None
        input_207 = torch.nn.functional.adaptive_avg_pool2d(x_109, 2)
        x_122 = torch.conv2d(
            input_207,
            l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_207 = l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_conv_parameters_weight_ = (None)
        x_123 = torch.nn.functional.batch_norm(
            x_122,
            l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_122 = l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_bias_ = (None)
        x_124 = torch.nn.functional.relu(x_123, inplace=True)
        x_123 = None
        upsampled_ppm_out_1 = torch.nn.functional.interpolate(
            x_124, (16, 16), None, "bilinear", False
        )
        x_124 = None
        input_208 = torch.nn.functional.adaptive_avg_pool2d(x_109, 3)
        x_125 = torch.conv2d(
            input_208,
            l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_208 = l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_conv_parameters_weight_ = (None)
        x_126 = torch.nn.functional.batch_norm(
            x_125,
            l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_125 = l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_bias_ = (None)
        x_127 = torch.nn.functional.relu(x_126, inplace=True)
        x_126 = None
        upsampled_ppm_out_2 = torch.nn.functional.interpolate(
            x_127, (16, 16), None, "bilinear", False
        )
        x_127 = None
        input_209 = torch.nn.functional.adaptive_avg_pool2d(x_109, 6)
        x_128 = torch.conv2d(
            input_209,
            l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_209 = l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_conv_parameters_weight_ = (None)
        x_129 = torch.nn.functional.batch_norm(
            x_128,
            l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_128 = l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_bias_ = (None)
        x_130 = torch.nn.functional.relu(x_129, inplace=True)
        x_129 = None
        upsampled_ppm_out_3 = torch.nn.functional.interpolate(
            x_130, (16, 16), None, "bilinear", False
        )
        x_130 = None
        psp_outs = torch.cat(
            [
                x_109,
                upsampled_ppm_out,
                upsampled_ppm_out_1,
                upsampled_ppm_out_2,
                upsampled_ppm_out_3,
            ],
            dim=1,
        )
        x_109 = (
            upsampled_ppm_out
        ) = upsampled_ppm_out_1 = upsampled_ppm_out_2 = upsampled_ppm_out_3 = None
        x_131 = torch.conv2d(
            psp_outs,
            l_self_modules_decode_head_modules_bottleneck_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        psp_outs = l_self_modules_decode_head_modules_bottleneck_modules_conv_parameters_weight_ = (None)
        x_132 = torch.nn.functional.batch_norm(
            x_131,
            l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_131 = l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_var_ = (
            l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_weight_
        ) = (
            l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_bias_
        ) = None
        x_133 = torch.nn.functional.relu(x_132, inplace=True)
        x_132 = None
        interpolate_4 = torch.nn.functional.interpolate(
            x_133, (32, 32), None, "bilinear", False
        )
        add_127 = x_118 + interpolate_4
        x_118 = interpolate_4 = None
        interpolate_5 = torch.nn.functional.interpolate(
            add_127, (64, 64), None, "bilinear", False
        )
        add_128 = x_115 + interpolate_5
        x_115 = interpolate_5 = None
        interpolate_6 = torch.nn.functional.interpolate(
            add_128, (128, 128), None, "bilinear", False
        )
        add_129 = x_112 + interpolate_6
        x_112 = interpolate_6 = None
        x_134 = torch.conv2d(
            add_129,
            l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        add_129 = l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_conv_parameters_weight_ = (None)
        x_135 = torch.nn.functional.batch_norm(
            x_134,
            l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_134 = l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_bias_ = (None)
        x_136 = torch.nn.functional.relu(x_135, inplace=False)
        x_135 = None
        x_137 = torch.conv2d(
            add_128,
            l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        add_128 = l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_conv_parameters_weight_ = (None)
        x_138 = torch.nn.functional.batch_norm(
            x_137,
            l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_137 = l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_bias_ = (None)
        x_139 = torch.nn.functional.relu(x_138, inplace=False)
        x_138 = None
        x_140 = torch.conv2d(
            add_127,
            l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        add_127 = l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_conv_parameters_weight_ = (None)
        x_141 = torch.nn.functional.batch_norm(
            x_140,
            l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_140 = l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_bias_ = (None)
        x_142 = torch.nn.functional.relu(x_141, inplace=False)
        x_141 = None
        interpolate_7 = torch.nn.functional.interpolate(
            x_133, (128, 128), None, "bilinear", False
        )
        x_133 = None
        interpolate_8 = torch.nn.functional.interpolate(
            x_142, (128, 128), None, "bilinear", False
        )
        x_142 = None
        interpolate_9 = torch.nn.functional.interpolate(
            x_139, (128, 128), None, "bilinear", False
        )
        x_139 = None
        fpn_outs = torch.cat(
            [x_136, interpolate_9, interpolate_8, interpolate_7], dim=1
        )
        x_136 = interpolate_9 = interpolate_8 = interpolate_7 = None
        x_143 = torch.conv2d(
            fpn_outs,
            l_self_modules_decode_head_modules_fpn_bottleneck_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        fpn_outs = l_self_modules_decode_head_modules_fpn_bottleneck_modules_conv_parameters_weight_ = (None)
        x_144 = torch.nn.functional.batch_norm(
            x_143,
            l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_143 = l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_bias_ = (None)
        x_145 = torch.nn.functional.relu(x_144, inplace=True)
        x_144 = None
        feat = torch.nn.functional.dropout2d(x_145, 0.1, False, False)
        x_145 = None
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
