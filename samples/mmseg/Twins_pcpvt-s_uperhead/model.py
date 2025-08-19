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
        reshape_8 = x_26.reshape(1, 64, 64, -1)
        x_26 = None
        permute_1 = reshape_8.permute(0, 3, 1, 2)
        reshape_8 = None
        x_27 = permute_1.contiguous()
        permute_1 = None
        x_28 = torch.conv2d(
            x_27,
            l_self_modules_backbone_modules_patch_embeds_modules_2_modules_projection_parameters_weight_,
            l_self_modules_backbone_modules_patch_embeds_modules_2_modules_projection_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_patch_embeds_modules_2_modules_projection_parameters_weight_ = l_self_modules_backbone_modules_patch_embeds_modules_2_modules_projection_parameters_bias_ = (None)
        flatten_11 = x_28.flatten(2)
        x_28 = None
        x_29 = flatten_11.transpose(1, 2)
        flatten_11 = None
        x_30 = torch.nn.functional.layer_norm(
            x_29,
            (320,),
            l_self_modules_backbone_modules_patch_embeds_modules_2_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_patch_embeds_modules_2_modules_norm_parameters_bias_,
            1e-05,
        )
        x_29 = l_self_modules_backbone_modules_patch_embeds_modules_2_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_patch_embeds_modules_2_modules_norm_parameters_bias_ = (None)
        x_31 = torch.nn.functional.dropout(x_30, 0.0, False, False)
        x_30 = None
        x_q_14 = torch.nn.functional.layer_norm(
            x_31,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm1_parameters_bias_ = (None)
        transpose_42 = x_q_14.transpose(1, 2)
        x_kv_35 = transpose_42.reshape(1, 320, 32, 32)
        transpose_42 = None
        x_kv_36 = torch.conv2d(
            x_kv_35,
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_35 = l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_12 = x_kv_36.flatten(2)
        x_kv_36 = None
        transpose_43 = flatten_12.transpose(1, 2)
        flatten_12 = None
        x_kv_37 = transpose_43.contiguous()
        transpose_43 = None
        x_kv_38 = torch.nn.functional.layer_norm(
            x_kv_37,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_37 = l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_15 = x_q_14.transpose(0, 1)
        x_q_14 = None
        x_kv_39 = x_kv_38.transpose(0, 1)
        x_kv_38 = None
        multi_head_attention_forward_7 = torch.nn.functional.multi_head_attention_forward(
            x_q_15,
            x_kv_39,
            x_kv_39,
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
        x_q_15 = (
            x_kv_39
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_7 = multi_head_attention_forward_7[0]
        multi_head_attention_forward_7 = None
        out_7 = attn_output_7.transpose(0, 1)
        attn_output_7 = None
        dropout_24 = torch.nn.functional.dropout(out_7, 0.0, False, False)
        out_7 = None
        add_23 = 0.0 + dropout_24
        dropout_24 = None
        x_32 = x_31 + add_23
        x_31 = add_23 = None
        layer_norm_26 = torch.nn.functional.layer_norm(
            x_32,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm2_parameters_bias_ = (None)
        input_36 = torch._C._nn.linear(
            layer_norm_26,
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_26 = l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_37 = torch._C._nn.gelu(input_36, approximate="none")
        input_36 = None
        input_38 = torch.nn.functional.dropout(input_37, 0.0, False, False)
        input_37 = None
        input_39 = torch._C._nn.linear(
            input_38,
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_38 = l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_40 = torch.nn.functional.dropout(input_39, 0.0, False, False)
        input_39 = None
        x_33 = x_32 + input_40
        x_32 = input_40 = None
        transpose_47 = x_33.transpose(1, 2)
        x_33 = None
        cnn_feat_2 = transpose_47.view(1, 320, 32, 32)
        transpose_47 = None
        conv2d_13 = torch.conv2d(
            cnn_feat_2,
            l_self_modules_backbone_modules_position_encodings_modules_2_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_position_encodings_modules_2_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_position_encodings_modules_2_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_position_encodings_modules_2_modules_proj_parameters_bias_ = (None)
        x_34 = conv2d_13 + cnn_feat_2
        conv2d_13 = cnn_feat_2 = None
        flatten_13 = x_34.flatten(2)
        x_34 = None
        x_35 = flatten_13.transpose(1, 2)
        flatten_13 = None
        x_q_16 = torch.nn.functional.layer_norm(
            x_35,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_norm1_parameters_bias_ = (None)
        transpose_49 = x_q_16.transpose(1, 2)
        x_kv_40 = transpose_49.reshape(1, 320, 32, 32)
        transpose_49 = None
        x_kv_41 = torch.conv2d(
            x_kv_40,
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_40 = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_14 = x_kv_41.flatten(2)
        x_kv_41 = None
        transpose_50 = flatten_14.transpose(1, 2)
        flatten_14 = None
        x_kv_42 = transpose_50.contiguous()
        transpose_50 = None
        x_kv_43 = torch.nn.functional.layer_norm(
            x_kv_42,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_42 = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_17 = x_q_16.transpose(0, 1)
        x_q_16 = None
        x_kv_44 = x_kv_43.transpose(0, 1)
        x_kv_43 = None
        multi_head_attention_forward_8 = torch.nn.functional.multi_head_attention_forward(
            x_q_17,
            x_kv_44,
            x_kv_44,
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
        x_q_17 = (
            x_kv_44
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_8 = multi_head_attention_forward_8[0]
        multi_head_attention_forward_8 = None
        out_8 = attn_output_8.transpose(0, 1)
        attn_output_8 = None
        dropout_27 = torch.nn.functional.dropout(out_8, 0.0, False, False)
        out_8 = None
        add_27 = 0.0 + dropout_27
        dropout_27 = None
        x_36 = x_35 + add_27
        x_35 = add_27 = None
        layer_norm_29 = torch.nn.functional.layer_norm(
            x_36,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_norm2_parameters_bias_ = (None)
        input_41 = torch._C._nn.linear(
            layer_norm_29,
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_29 = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_42 = torch._C._nn.gelu(input_41, approximate="none")
        input_41 = None
        input_43 = torch.nn.functional.dropout(input_42, 0.0, False, False)
        input_42 = None
        input_44 = torch._C._nn.linear(
            input_43,
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_43 = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_45 = torch.nn.functional.dropout(input_44, 0.0, False, False)
        input_44 = None
        x_37 = x_36 + input_45
        x_36 = input_45 = None
        x_q_18 = torch.nn.functional.layer_norm(
            x_37,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_norm1_parameters_bias_ = (None)
        transpose_54 = x_q_18.transpose(1, 2)
        x_kv_45 = transpose_54.reshape(1, 320, 32, 32)
        transpose_54 = None
        x_kv_46 = torch.conv2d(
            x_kv_45,
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_45 = l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_15 = x_kv_46.flatten(2)
        x_kv_46 = None
        transpose_55 = flatten_15.transpose(1, 2)
        flatten_15 = None
        x_kv_47 = transpose_55.contiguous()
        transpose_55 = None
        x_kv_48 = torch.nn.functional.layer_norm(
            x_kv_47,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_47 = l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_19 = x_q_18.transpose(0, 1)
        x_q_18 = None
        x_kv_49 = x_kv_48.transpose(0, 1)
        x_kv_48 = None
        multi_head_attention_forward_9 = torch.nn.functional.multi_head_attention_forward(
            x_q_19,
            x_kv_49,
            x_kv_49,
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
        x_q_19 = (
            x_kv_49
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_9 = multi_head_attention_forward_9[0]
        multi_head_attention_forward_9 = None
        out_9 = attn_output_9.transpose(0, 1)
        attn_output_9 = None
        dropout_30 = torch.nn.functional.dropout(out_9, 0.0, False, False)
        out_9 = None
        add_30 = 0.0 + dropout_30
        dropout_30 = None
        x_38 = x_37 + add_30
        x_37 = add_30 = None
        layer_norm_32 = torch.nn.functional.layer_norm(
            x_38,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_norm2_parameters_bias_ = (None)
        input_46 = torch._C._nn.linear(
            layer_norm_32,
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_32 = l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_47 = torch._C._nn.gelu(input_46, approximate="none")
        input_46 = None
        input_48 = torch.nn.functional.dropout(input_47, 0.0, False, False)
        input_47 = None
        input_49 = torch._C._nn.linear(
            input_48,
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_48 = l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_50 = torch.nn.functional.dropout(input_49, 0.0, False, False)
        input_49 = None
        x_39 = x_38 + input_50
        x_38 = input_50 = None
        x_q_20 = torch.nn.functional.layer_norm(
            x_39,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_norm1_parameters_bias_ = (None)
        transpose_59 = x_q_20.transpose(1, 2)
        x_kv_50 = transpose_59.reshape(1, 320, 32, 32)
        transpose_59 = None
        x_kv_51 = torch.conv2d(
            x_kv_50,
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_50 = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_16 = x_kv_51.flatten(2)
        x_kv_51 = None
        transpose_60 = flatten_16.transpose(1, 2)
        flatten_16 = None
        x_kv_52 = transpose_60.contiguous()
        transpose_60 = None
        x_kv_53 = torch.nn.functional.layer_norm(
            x_kv_52,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_52 = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_21 = x_q_20.transpose(0, 1)
        x_q_20 = None
        x_kv_54 = x_kv_53.transpose(0, 1)
        x_kv_53 = None
        multi_head_attention_forward_10 = torch.nn.functional.multi_head_attention_forward(
            x_q_21,
            x_kv_54,
            x_kv_54,
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
        x_q_21 = (
            x_kv_54
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_10 = multi_head_attention_forward_10[0]
        multi_head_attention_forward_10 = None
        out_10 = attn_output_10.transpose(0, 1)
        attn_output_10 = None
        dropout_33 = torch.nn.functional.dropout(out_10, 0.0, False, False)
        out_10 = None
        add_33 = 0.0 + dropout_33
        dropout_33 = None
        x_40 = x_39 + add_33
        x_39 = add_33 = None
        layer_norm_35 = torch.nn.functional.layer_norm(
            x_40,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_norm2_parameters_bias_ = (None)
        input_51 = torch._C._nn.linear(
            layer_norm_35,
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_35 = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_52 = torch._C._nn.gelu(input_51, approximate="none")
        input_51 = None
        input_53 = torch.nn.functional.dropout(input_52, 0.0, False, False)
        input_52 = None
        input_54 = torch._C._nn.linear(
            input_53,
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_53 = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_55 = torch.nn.functional.dropout(input_54, 0.0, False, False)
        input_54 = None
        x_41 = x_40 + input_55
        x_40 = input_55 = None
        x_q_22 = torch.nn.functional.layer_norm(
            x_41,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_norm1_parameters_bias_ = (None)
        transpose_64 = x_q_22.transpose(1, 2)
        x_kv_55 = transpose_64.reshape(1, 320, 32, 32)
        transpose_64 = None
        x_kv_56 = torch.conv2d(
            x_kv_55,
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_55 = l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_17 = x_kv_56.flatten(2)
        x_kv_56 = None
        transpose_65 = flatten_17.transpose(1, 2)
        flatten_17 = None
        x_kv_57 = transpose_65.contiguous()
        transpose_65 = None
        x_kv_58 = torch.nn.functional.layer_norm(
            x_kv_57,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_57 = l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_norm_parameters_bias_ = (None)
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
        x_q_23 = (
            x_kv_59
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_11 = multi_head_attention_forward_11[0]
        multi_head_attention_forward_11 = None
        out_11 = attn_output_11.transpose(0, 1)
        attn_output_11 = None
        dropout_36 = torch.nn.functional.dropout(out_11, 0.0, False, False)
        out_11 = None
        add_36 = 0.0 + dropout_36
        dropout_36 = None
        x_42 = x_41 + add_36
        x_41 = add_36 = None
        layer_norm_38 = torch.nn.functional.layer_norm(
            x_42,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_norm2_parameters_bias_ = (None)
        input_56 = torch._C._nn.linear(
            layer_norm_38,
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_38 = l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_57 = torch._C._nn.gelu(input_56, approximate="none")
        input_56 = None
        input_58 = torch.nn.functional.dropout(input_57, 0.0, False, False)
        input_57 = None
        input_59 = torch._C._nn.linear(
            input_58,
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_58 = l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_60 = torch.nn.functional.dropout(input_59, 0.0, False, False)
        input_59 = None
        x_43 = x_42 + input_60
        x_42 = input_60 = None
        x_q_24 = torch.nn.functional.layer_norm(
            x_43,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_norm1_parameters_bias_ = (None)
        transpose_69 = x_q_24.transpose(1, 2)
        x_kv_60 = transpose_69.reshape(1, 320, 32, 32)
        transpose_69 = None
        x_kv_61 = torch.conv2d(
            x_kv_60,
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_60 = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_18 = x_kv_61.flatten(2)
        x_kv_61 = None
        transpose_70 = flatten_18.transpose(1, 2)
        flatten_18 = None
        x_kv_62 = transpose_70.contiguous()
        transpose_70 = None
        x_kv_63 = torch.nn.functional.layer_norm(
            x_kv_62,
            (320,),
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_62 = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_norm_parameters_bias_ = (None)
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
        x_q_25 = (
            x_kv_64
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
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
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_norm2_parameters_bias_ = (None)
        input_61 = torch._C._nn.linear(
            layer_norm_41,
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_41 = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_62 = torch._C._nn.gelu(input_61, approximate="none")
        input_61 = None
        input_63 = torch.nn.functional.dropout(input_62, 0.0, False, False)
        input_62 = None
        input_64 = torch._C._nn.linear(
            input_63,
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_63 = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_65 = torch.nn.functional.dropout(input_64, 0.0, False, False)
        input_64 = None
        x_45 = x_44 + input_65
        x_44 = input_65 = None
        reshape_15 = x_45.reshape(1, 32, 32, -1)
        x_45 = None
        permute_2 = reshape_15.permute(0, 3, 1, 2)
        reshape_15 = None
        x_46 = permute_2.contiguous()
        permute_2 = None
        x_47 = torch.conv2d(
            x_46,
            l_self_modules_backbone_modules_patch_embeds_modules_3_modules_projection_parameters_weight_,
            l_self_modules_backbone_modules_patch_embeds_modules_3_modules_projection_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_patch_embeds_modules_3_modules_projection_parameters_weight_ = l_self_modules_backbone_modules_patch_embeds_modules_3_modules_projection_parameters_bias_ = (None)
        flatten_19 = x_47.flatten(2)
        x_47 = None
        x_48 = flatten_19.transpose(1, 2)
        flatten_19 = None
        x_49 = torch.nn.functional.layer_norm(
            x_48,
            (512,),
            l_self_modules_backbone_modules_patch_embeds_modules_3_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_patch_embeds_modules_3_modules_norm_parameters_bias_,
            1e-05,
        )
        x_48 = l_self_modules_backbone_modules_patch_embeds_modules_3_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_patch_embeds_modules_3_modules_norm_parameters_bias_ = (None)
        x_50 = torch.nn.functional.dropout(x_49, 0.0, False, False)
        x_49 = None
        x_q_26 = torch.nn.functional.layer_norm(
            x_50,
            (512,),
            l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm1_parameters_bias_ = (None)
        x_q_27 = x_q_26.transpose(0, 1)
        x_kv_65 = x_q_26.transpose(0, 1)
        x_q_26 = None
        multi_head_attention_forward_13 = torch.nn.functional.multi_head_attention_forward(
            x_q_27,
            x_kv_65,
            x_kv_65,
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
        x_q_27 = (
            x_kv_65
        ) = l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_13 = multi_head_attention_forward_13[0]
        multi_head_attention_forward_13 = None
        out_13 = attn_output_13.transpose(0, 1)
        attn_output_13 = None
        dropout_43 = torch.nn.functional.dropout(out_13, 0.0, False, False)
        out_13 = None
        add_42 = 0.0 + dropout_43
        dropout_43 = None
        x_51 = x_50 + add_42
        x_50 = add_42 = None
        layer_norm_44 = torch.nn.functional.layer_norm(
            x_51,
            (512,),
            l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm2_parameters_bias_ = (None)
        input_66 = torch._C._nn.linear(
            layer_norm_44,
            l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_44 = l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_67 = torch._C._nn.gelu(input_66, approximate="none")
        input_66 = None
        input_68 = torch.nn.functional.dropout(input_67, 0.0, False, False)
        input_67 = None
        input_69 = torch._C._nn.linear(
            input_68,
            l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_68 = l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_70 = torch.nn.functional.dropout(input_69, 0.0, False, False)
        input_69 = None
        x_52 = x_51 + input_70
        x_51 = input_70 = None
        transpose_78 = x_52.transpose(1, 2)
        x_52 = None
        cnn_feat_3 = transpose_78.view(1, 512, 16, 16)
        transpose_78 = None
        conv2d_20 = torch.conv2d(
            cnn_feat_3,
            l_self_modules_backbone_modules_position_encodings_modules_3_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_position_encodings_modules_3_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        l_self_modules_backbone_modules_position_encodings_modules_3_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_position_encodings_modules_3_modules_proj_parameters_bias_ = (None)
        x_53 = conv2d_20 + cnn_feat_3
        conv2d_20 = cnn_feat_3 = None
        flatten_20 = x_53.flatten(2)
        x_53 = None
        x_54 = flatten_20.transpose(1, 2)
        flatten_20 = None
        x_q_28 = torch.nn.functional.layer_norm(
            x_54,
            (512,),
            l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_norm1_parameters_bias_ = (None)
        x_q_29 = x_q_28.transpose(0, 1)
        x_kv_66 = x_q_28.transpose(0, 1)
        x_q_28 = None
        multi_head_attention_forward_14 = torch.nn.functional.multi_head_attention_forward(
            x_q_29,
            x_kv_66,
            x_kv_66,
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
        x_q_29 = (
            x_kv_66
        ) = l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_14 = multi_head_attention_forward_14[0]
        multi_head_attention_forward_14 = None
        out_14 = attn_output_14.transpose(0, 1)
        attn_output_14 = None
        dropout_46 = torch.nn.functional.dropout(out_14, 0.0, False, False)
        out_14 = None
        add_46 = 0.0 + dropout_46
        dropout_46 = None
        x_55 = x_54 + add_46
        x_54 = add_46 = None
        layer_norm_46 = torch.nn.functional.layer_norm(
            x_55,
            (512,),
            l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_norm2_parameters_bias_ = (None)
        input_71 = torch._C._nn.linear(
            layer_norm_46,
            l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_46 = l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_72 = torch._C._nn.gelu(input_71, approximate="none")
        input_71 = None
        input_73 = torch.nn.functional.dropout(input_72, 0.0, False, False)
        input_72 = None
        input_74 = torch._C._nn.linear(
            input_73,
            l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_73 = l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_75 = torch.nn.functional.dropout(input_74, 0.0, False, False)
        input_74 = None
        x_56 = x_55 + input_75
        x_55 = input_75 = None
        x_q_30 = torch.nn.functional.layer_norm(
            x_56,
            (512,),
            l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_norm1_parameters_bias_ = (None)
        x_q_31 = x_q_30.transpose(0, 1)
        x_kv_67 = x_q_30.transpose(0, 1)
        x_q_30 = None
        multi_head_attention_forward_15 = torch.nn.functional.multi_head_attention_forward(
            x_q_31,
            x_kv_67,
            x_kv_67,
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
        x_q_31 = (
            x_kv_67
        ) = l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_15 = multi_head_attention_forward_15[0]
        multi_head_attention_forward_15 = None
        out_15 = attn_output_15.transpose(0, 1)
        attn_output_15 = None
        dropout_49 = torch.nn.functional.dropout(out_15, 0.0, False, False)
        out_15 = None
        add_49 = 0.0 + dropout_49
        dropout_49 = None
        x_57 = x_56 + add_49
        x_56 = add_49 = None
        layer_norm_48 = torch.nn.functional.layer_norm(
            x_57,
            (512,),
            l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_norm2_parameters_bias_ = (None)
        input_76 = torch._C._nn.linear(
            layer_norm_48,
            l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_48 = l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_77 = torch._C._nn.gelu(input_76, approximate="none")
        input_76 = None
        input_78 = torch.nn.functional.dropout(input_77, 0.0, False, False)
        input_77 = None
        input_79 = torch._C._nn.linear(
            input_78,
            l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_78 = l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_80 = torch.nn.functional.dropout(input_79, 0.0, False, False)
        input_79 = None
        x_58 = x_57 + input_80
        x_57 = input_80 = None
        reshape_16 = x_58.reshape(1, 16, 16, -1)
        x_58 = None
        permute_3 = reshape_16.permute(0, 3, 1, 2)
        reshape_16 = None
        x_59 = permute_3.contiguous()
        permute_3 = None
        x_60 = torch.conv2d(
            x_12,
            l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_12 = l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_conv_parameters_weight_ = (None)
        x_61 = torch.nn.functional.batch_norm(
            x_60,
            l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_60 = l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_bias_ = (None)
        x_62 = torch.nn.functional.relu(x_61, inplace=False)
        x_61 = None
        x_63 = torch.conv2d(
            x_27,
            l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_27 = l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_conv_parameters_weight_ = (None)
        x_64 = torch.nn.functional.batch_norm(
            x_63,
            l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_63 = l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_bias_ = (None)
        x_65 = torch.nn.functional.relu(x_64, inplace=False)
        x_64 = None
        x_66 = torch.conv2d(
            x_46,
            l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_46 = l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_conv_parameters_weight_ = (None)
        x_67 = torch.nn.functional.batch_norm(
            x_66,
            l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_66 = l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_bias_ = (None)
        x_68 = torch.nn.functional.relu(x_67, inplace=False)
        x_67 = None
        input_81 = torch.nn.functional.adaptive_avg_pool2d(x_59, 1)
        x_69 = torch.conv2d(
            input_81,
            l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_81 = l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_conv_parameters_weight_ = (None)
        x_70 = torch.nn.functional.batch_norm(
            x_69,
            l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_69 = l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_bias_ = (None)
        x_71 = torch.nn.functional.relu(x_70, inplace=True)
        x_70 = None
        upsampled_ppm_out = torch.nn.functional.interpolate(
            x_71, (16, 16), None, "bilinear", False
        )
        x_71 = None
        input_82 = torch.nn.functional.adaptive_avg_pool2d(x_59, 2)
        x_72 = torch.conv2d(
            input_82,
            l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_82 = l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_conv_parameters_weight_ = (None)
        x_73 = torch.nn.functional.batch_norm(
            x_72,
            l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_72 = l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_bias_ = (None)
        x_74 = torch.nn.functional.relu(x_73, inplace=True)
        x_73 = None
        upsampled_ppm_out_1 = torch.nn.functional.interpolate(
            x_74, (16, 16), None, "bilinear", False
        )
        x_74 = None
        input_83 = torch.nn.functional.adaptive_avg_pool2d(x_59, 3)
        x_75 = torch.conv2d(
            input_83,
            l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_83 = l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_conv_parameters_weight_ = (None)
        x_76 = torch.nn.functional.batch_norm(
            x_75,
            l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_75 = l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_bias_ = (None)
        x_77 = torch.nn.functional.relu(x_76, inplace=True)
        x_76 = None
        upsampled_ppm_out_2 = torch.nn.functional.interpolate(
            x_77, (16, 16), None, "bilinear", False
        )
        x_77 = None
        input_84 = torch.nn.functional.adaptive_avg_pool2d(x_59, 6)
        x_78 = torch.conv2d(
            input_84,
            l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_84 = l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_conv_parameters_weight_ = (None)
        x_79 = torch.nn.functional.batch_norm(
            x_78,
            l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_78 = l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_bias_ = (None)
        x_80 = torch.nn.functional.relu(x_79, inplace=True)
        x_79 = None
        upsampled_ppm_out_3 = torch.nn.functional.interpolate(
            x_80, (16, 16), None, "bilinear", False
        )
        x_80 = None
        psp_outs = torch.cat(
            [
                x_59,
                upsampled_ppm_out,
                upsampled_ppm_out_1,
                upsampled_ppm_out_2,
                upsampled_ppm_out_3,
            ],
            dim=1,
        )
        x_59 = (
            upsampled_ppm_out
        ) = upsampled_ppm_out_1 = upsampled_ppm_out_2 = upsampled_ppm_out_3 = None
        x_81 = torch.conv2d(
            psp_outs,
            l_self_modules_decode_head_modules_bottleneck_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        psp_outs = l_self_modules_decode_head_modules_bottleneck_modules_conv_parameters_weight_ = (None)
        x_82 = torch.nn.functional.batch_norm(
            x_81,
            l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_81 = l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_var_ = (
            l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_weight_
        ) = (
            l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_bias_
        ) = None
        x_83 = torch.nn.functional.relu(x_82, inplace=True)
        x_82 = None
        interpolate_4 = torch.nn.functional.interpolate(
            x_83, (32, 32), None, "bilinear", False
        )
        add_52 = x_68 + interpolate_4
        x_68 = interpolate_4 = None
        interpolate_5 = torch.nn.functional.interpolate(
            add_52, (64, 64), None, "bilinear", False
        )
        add_53 = x_65 + interpolate_5
        x_65 = interpolate_5 = None
        interpolate_6 = torch.nn.functional.interpolate(
            add_53, (128, 128), None, "bilinear", False
        )
        add_54 = x_62 + interpolate_6
        x_62 = interpolate_6 = None
        x_84 = torch.conv2d(
            add_54,
            l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        add_54 = l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_conv_parameters_weight_ = (None)
        x_85 = torch.nn.functional.batch_norm(
            x_84,
            l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_84 = l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_bias_ = (None)
        x_86 = torch.nn.functional.relu(x_85, inplace=False)
        x_85 = None
        x_87 = torch.conv2d(
            add_53,
            l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        add_53 = l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_conv_parameters_weight_ = (None)
        x_88 = torch.nn.functional.batch_norm(
            x_87,
            l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_87 = l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_bias_ = (None)
        x_89 = torch.nn.functional.relu(x_88, inplace=False)
        x_88 = None
        x_90 = torch.conv2d(
            add_52,
            l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        add_52 = l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_conv_parameters_weight_ = (None)
        x_91 = torch.nn.functional.batch_norm(
            x_90,
            l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_90 = l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_bias_ = (None)
        x_92 = torch.nn.functional.relu(x_91, inplace=False)
        x_91 = None
        interpolate_7 = torch.nn.functional.interpolate(
            x_83, (128, 128), None, "bilinear", False
        )
        x_83 = None
        interpolate_8 = torch.nn.functional.interpolate(
            x_92, (128, 128), None, "bilinear", False
        )
        x_92 = None
        interpolate_9 = torch.nn.functional.interpolate(
            x_89, (128, 128), None, "bilinear", False
        )
        x_89 = None
        fpn_outs = torch.cat([x_86, interpolate_9, interpolate_8, interpolate_7], dim=1)
        x_86 = interpolate_9 = interpolate_8 = interpolate_7 = None
        x_93 = torch.conv2d(
            fpn_outs,
            l_self_modules_decode_head_modules_fpn_bottleneck_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        fpn_outs = l_self_modules_decode_head_modules_fpn_bottleneck_modules_conv_parameters_weight_ = (None)
        x_94 = torch.nn.functional.batch_norm(
            x_93,
            l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_93 = l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_bias_ = (None)
        x_95 = torch.nn.functional.relu(x_94, inplace=True)
        x_94 = None
        feat = torch.nn.functional.dropout2d(x_95, 0.1, False, False)
        x_95 = None
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
