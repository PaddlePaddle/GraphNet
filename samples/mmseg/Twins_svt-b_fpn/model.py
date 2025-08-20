import torch

from torch import device


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
        L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_norm_list_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_norm_list_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embeds_modules_1_modules_projection_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embeds_modules_1_modules_projection_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embeds_modules_1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embeds_modules_1_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_norm_list_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_norm_list_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embeds_modules_2_modules_projection_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embeds_modules_2_modules_projection_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embeds_modules_2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embeds_modules_2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_norm_list_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_norm_list_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embeds_modules_3_modules_projection_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embeds_modules_3_modules_projection_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embeds_modules_3_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embeds_modules_3_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_norm_list_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_norm_list_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_lateral_convs_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_lateral_convs_modules_0_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_lateral_convs_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_lateral_convs_modules_1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_lateral_convs_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_lateral_convs_modules_2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_lateral_convs_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_lateral_convs_modules_3_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_convs_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_convs_modules_0_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_convs_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_convs_modules_1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_convs_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_convs_modules_2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_convs_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_convs_modules_3_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_
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
        l_self_modules_backbone_modules_norm_list_modules_0_parameters_weight_ = (
            L_self_modules_backbone_modules_norm_list_modules_0_parameters_weight_
        )
        l_self_modules_backbone_modules_norm_list_modules_0_parameters_bias_ = (
            L_self_modules_backbone_modules_norm_list_modules_0_parameters_bias_
        )
        l_self_modules_backbone_modules_patch_embeds_modules_1_modules_projection_parameters_weight_ = L_self_modules_backbone_modules_patch_embeds_modules_1_modules_projection_parameters_weight_
        l_self_modules_backbone_modules_patch_embeds_modules_1_modules_projection_parameters_bias_ = L_self_modules_backbone_modules_patch_embeds_modules_1_modules_projection_parameters_bias_
        l_self_modules_backbone_modules_patch_embeds_modules_1_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_patch_embeds_modules_1_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_patch_embeds_modules_1_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_patch_embeds_modules_1_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_
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
        l_self_modules_backbone_modules_norm_list_modules_1_parameters_weight_ = (
            L_self_modules_backbone_modules_norm_list_modules_1_parameters_weight_
        )
        l_self_modules_backbone_modules_norm_list_modules_1_parameters_bias_ = (
            L_self_modules_backbone_modules_norm_list_modules_1_parameters_bias_
        )
        l_self_modules_backbone_modules_patch_embeds_modules_2_modules_projection_parameters_weight_ = L_self_modules_backbone_modules_patch_embeds_modules_2_modules_projection_parameters_weight_
        l_self_modules_backbone_modules_patch_embeds_modules_2_modules_projection_parameters_bias_ = L_self_modules_backbone_modules_patch_embeds_modules_2_modules_projection_parameters_bias_
        l_self_modules_backbone_modules_patch_embeds_modules_2_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_patch_embeds_modules_2_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_patch_embeds_modules_2_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_patch_embeds_modules_2_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_proj_parameters_bias_
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
        l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_proj_parameters_bias_
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
        l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_proj_parameters_bias_
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
        l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_proj_parameters_bias_
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
        l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_proj_parameters_bias_
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
        l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_proj_parameters_bias_
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
        l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_proj_parameters_bias_
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
        l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_proj_parameters_bias_
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
        l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_proj_parameters_bias_
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
        l_self_modules_backbone_modules_norm_list_modules_2_parameters_weight_ = (
            L_self_modules_backbone_modules_norm_list_modules_2_parameters_weight_
        )
        l_self_modules_backbone_modules_norm_list_modules_2_parameters_bias_ = (
            L_self_modules_backbone_modules_norm_list_modules_2_parameters_bias_
        )
        l_self_modules_backbone_modules_patch_embeds_modules_3_modules_projection_parameters_weight_ = L_self_modules_backbone_modules_patch_embeds_modules_3_modules_projection_parameters_weight_
        l_self_modules_backbone_modules_patch_embeds_modules_3_modules_projection_parameters_bias_ = L_self_modules_backbone_modules_patch_embeds_modules_3_modules_projection_parameters_bias_
        l_self_modules_backbone_modules_patch_embeds_modules_3_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_patch_embeds_modules_3_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_patch_embeds_modules_3_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_patch_embeds_modules_3_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_proj_parameters_bias_
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
        l_self_modules_backbone_modules_norm_list_modules_3_parameters_weight_ = (
            L_self_modules_backbone_modules_norm_list_modules_3_parameters_weight_
        )
        l_self_modules_backbone_modules_norm_list_modules_3_parameters_bias_ = (
            L_self_modules_backbone_modules_norm_list_modules_3_parameters_bias_
        )
        l_self_modules_neck_modules_lateral_convs_modules_0_modules_conv_parameters_weight_ = L_self_modules_neck_modules_lateral_convs_modules_0_modules_conv_parameters_weight_
        l_self_modules_neck_modules_lateral_convs_modules_0_modules_conv_parameters_bias_ = L_self_modules_neck_modules_lateral_convs_modules_0_modules_conv_parameters_bias_
        l_self_modules_neck_modules_lateral_convs_modules_1_modules_conv_parameters_weight_ = L_self_modules_neck_modules_lateral_convs_modules_1_modules_conv_parameters_weight_
        l_self_modules_neck_modules_lateral_convs_modules_1_modules_conv_parameters_bias_ = L_self_modules_neck_modules_lateral_convs_modules_1_modules_conv_parameters_bias_
        l_self_modules_neck_modules_lateral_convs_modules_2_modules_conv_parameters_weight_ = L_self_modules_neck_modules_lateral_convs_modules_2_modules_conv_parameters_weight_
        l_self_modules_neck_modules_lateral_convs_modules_2_modules_conv_parameters_bias_ = L_self_modules_neck_modules_lateral_convs_modules_2_modules_conv_parameters_bias_
        l_self_modules_neck_modules_lateral_convs_modules_3_modules_conv_parameters_weight_ = L_self_modules_neck_modules_lateral_convs_modules_3_modules_conv_parameters_weight_
        l_self_modules_neck_modules_lateral_convs_modules_3_modules_conv_parameters_bias_ = L_self_modules_neck_modules_lateral_convs_modules_3_modules_conv_parameters_bias_
        l_self_modules_neck_modules_fpn_convs_modules_0_modules_conv_parameters_weight_ = L_self_modules_neck_modules_fpn_convs_modules_0_modules_conv_parameters_weight_
        l_self_modules_neck_modules_fpn_convs_modules_0_modules_conv_parameters_bias_ = L_self_modules_neck_modules_fpn_convs_modules_0_modules_conv_parameters_bias_
        l_self_modules_neck_modules_fpn_convs_modules_1_modules_conv_parameters_weight_ = L_self_modules_neck_modules_fpn_convs_modules_1_modules_conv_parameters_weight_
        l_self_modules_neck_modules_fpn_convs_modules_1_modules_conv_parameters_bias_ = L_self_modules_neck_modules_fpn_convs_modules_1_modules_conv_parameters_bias_
        l_self_modules_neck_modules_fpn_convs_modules_2_modules_conv_parameters_weight_ = L_self_modules_neck_modules_fpn_convs_modules_2_modules_conv_parameters_weight_
        l_self_modules_neck_modules_fpn_convs_modules_2_modules_conv_parameters_bias_ = L_self_modules_neck_modules_fpn_convs_modules_2_modules_conv_parameters_bias_
        l_self_modules_neck_modules_fpn_convs_modules_3_modules_conv_parameters_weight_ = L_self_modules_neck_modules_fpn_convs_modules_3_modules_conv_parameters_weight_
        l_self_modules_neck_modules_fpn_convs_modules_3_modules_conv_parameters_bias_ = L_self_modules_neck_modules_fpn_convs_modules_3_modules_conv_parameters_bias_
        l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_parameters_bias_
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
            (96,),
            l_self_modules_backbone_modules_patch_embeds_modules_0_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_patch_embeds_modules_0_modules_norm_parameters_bias_,
            1e-05,
        )
        x_1 = l_self_modules_backbone_modules_patch_embeds_modules_0_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_patch_embeds_modules_0_modules_norm_parameters_bias_ = (None)
        x_3 = torch.nn.functional.dropout(x_2, 0.0, False, False)
        x_2 = None
        layer_norm_1 = torch.nn.functional.layer_norm(
            x_3,
            (96,),
            l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_norm1_parameters_bias_ = (None)
        x_4 = layer_norm_1.view(1, 128, 128, 96)
        layer_norm_1 = None
        x_5 = torch._C._nn.pad(x_4, (0, 0, 0, 5, 0, 5), "constant", None)
        x_4 = None
        mask = torch.zeros((1, 133, 133), device=device(type="cuda", index=0))
        getitem = mask[
            (slice(None, None, None), slice(-5, None, None), slice(None, None, None))
        ]
        fill_ = getitem.fill_(1)
        getitem = fill_ = None
        getitem_1 = mask[
            (slice(None, None, None), slice(None, None, None), slice(-5, None, None))
        ]
        fill__1 = getitem_1.fill_(1)
        getitem_1 = fill__1 = None
        reshape = x_5.reshape(1, 19, 7, 19, 7, 96)
        x_5 = None
        x_6 = reshape.transpose(2, 3)
        reshape = None
        reshape_1 = mask.reshape(1, 19, 7, 19, 7)
        mask = None
        transpose_2 = reshape_1.transpose(2, 3)
        reshape_1 = None
        mask_1 = transpose_2.reshape(1, 361, 49)
        transpose_2 = None
        unsqueeze = mask_1.unsqueeze(2)
        unsqueeze_1 = mask_1.unsqueeze(3)
        mask_1 = None
        attn_mask = unsqueeze - unsqueeze_1
        unsqueeze = unsqueeze_1 = None
        ne = attn_mask != 0
        masked_fill = attn_mask.masked_fill(ne, -1000.0)
        ne = None
        eq = attn_mask == 0
        attn_mask = None
        attn_mask_1 = masked_fill.masked_fill(eq, 0.0)
        masked_fill = eq = None
        linear = torch._C._nn.linear(
            x_6,
            l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        x_6 = l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_3 = linear.reshape(1, 361, 49, 3, 3, 32)
        linear = None
        qkv = reshape_3.permute(3, 0, 1, 4, 2, 5)
        reshape_3 = None
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]
        qkv = None
        transpose_3 = k.transpose(-2, -1)
        k = None
        matmul = q @ transpose_3
        q = transpose_3 = None
        attn = matmul * 0.1767766952966369
        matmul = None
        unsqueeze_2 = attn_mask_1.unsqueeze(2)
        attn_mask_1 = None
        attn_1 = attn + unsqueeze_2
        attn = unsqueeze_2 = None
        attn_2 = attn_1.softmax(dim=-1)
        attn_1 = None
        attn_3 = torch.nn.functional.dropout(attn_2, 0.0, False, False)
        attn_2 = None
        matmul_1 = attn_3 @ v
        attn_3 = v = None
        transpose_4 = matmul_1.transpose(2, 3)
        matmul_1 = None
        attn_4 = transpose_4.reshape(1, 19, 19, 7, 7, 96)
        transpose_4 = None
        transpose_5 = attn_4.transpose(2, 3)
        attn_4 = None
        x_7 = transpose_5.reshape(1, 133, 133, 96)
        transpose_5 = None
        getitem_5 = x_7[
            (
                slice(None, None, None),
                slice(None, 128, None),
                slice(None, 128, None),
                slice(None, None, None),
            )
        ]
        x_7 = None
        x_8 = getitem_5.contiguous()
        getitem_5 = None
        x_9 = x_8.reshape(1, 16384, 96)
        x_8 = None
        x_10 = torch._C._nn.linear(
            x_9,
            l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_9 = l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_11 = torch.nn.functional.dropout(x_10, 0.0, False, False)
        x_10 = None
        x_12 = x_3 + x_11
        x_3 = x_11 = None
        layer_norm_2 = torch.nn.functional.layer_norm(
            x_12,
            (96,),
            l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_norm2_parameters_bias_ = (None)
        input_1 = torch._C._nn.linear(
            layer_norm_2,
            l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_2 = l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
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
        x_13 = x_12 + input_5
        x_12 = input_5 = None
        transpose_6 = x_13.transpose(1, 2)
        x_13 = None
        cnn_feat = transpose_6.view(1, 96, 128, 128)
        transpose_6 = None
        conv2d_1 = torch.conv2d(
            cnn_feat,
            l_self_modules_backbone_modules_position_encodings_modules_0_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_position_encodings_modules_0_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            96,
        )
        l_self_modules_backbone_modules_position_encodings_modules_0_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_position_encodings_modules_0_modules_proj_parameters_bias_ = (None)
        x_14 = conv2d_1 + cnn_feat
        conv2d_1 = cnn_feat = None
        flatten_1 = x_14.flatten(2)
        x_14 = None
        x_15 = flatten_1.transpose(1, 2)
        flatten_1 = None
        x_q = torch.nn.functional.layer_norm(
            x_15,
            (96,),
            l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_norm1_parameters_bias_ = (None)
        transpose_8 = x_q.transpose(1, 2)
        x_kv = transpose_8.reshape(1, 96, 128, 128)
        transpose_8 = None
        x_kv_1 = torch.conv2d(
            x_kv,
            l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_sr_parameters_bias_,
            (8, 8),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv = l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_2 = x_kv_1.flatten(2)
        x_kv_1 = None
        transpose_9 = flatten_2.transpose(1, 2)
        flatten_2 = None
        x_kv_2 = transpose_9.contiguous()
        transpose_9 = None
        x_kv_3 = torch.nn.functional.layer_norm(
            x_kv_2,
            (96,),
            l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_2 = l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_1 = x_q.transpose(0, 1)
        x_q = None
        x_kv_4 = x_kv_3.transpose(0, 1)
        x_kv_3 = None
        multi_head_attention_forward = torch.nn.functional.multi_head_attention_forward(
            x_q_1,
            x_kv_4,
            x_kv_4,
            96,
            3,
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
        x_q_1 = (
            x_kv_4
        ) = l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output = multi_head_attention_forward[0]
        multi_head_attention_forward = None
        out = attn_output.transpose(0, 1)
        attn_output = None
        dropout_5 = torch.nn.functional.dropout(out, 0.0, False, False)
        out = None
        add_4 = 0.0 + dropout_5
        dropout_5 = None
        x_16 = x_15 + add_4
        x_15 = add_4 = None
        layer_norm_5 = torch.nn.functional.layer_norm(
            x_16,
            (96,),
            l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_norm2_parameters_bias_ = (None)
        input_6 = torch._C._nn.linear(
            layer_norm_5,
            l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_5 = l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
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
        x_17 = x_16 + input_10
        x_16 = input_10 = None
        x_18 = torch.nn.functional.layer_norm(
            x_17,
            (96,),
            l_self_modules_backbone_modules_norm_list_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_norm_list_modules_0_parameters_bias_,
            1e-05,
        )
        x_17 = (
            l_self_modules_backbone_modules_norm_list_modules_0_parameters_weight_
        ) = l_self_modules_backbone_modules_norm_list_modules_0_parameters_bias_ = None
        reshape_8 = x_18.reshape(1, 128, 128, -1)
        x_18 = None
        permute_1 = reshape_8.permute(0, 3, 1, 2)
        reshape_8 = None
        x_19 = permute_1.contiguous()
        permute_1 = None
        x_20 = torch.conv2d(
            x_19,
            l_self_modules_backbone_modules_patch_embeds_modules_1_modules_projection_parameters_weight_,
            l_self_modules_backbone_modules_patch_embeds_modules_1_modules_projection_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_patch_embeds_modules_1_modules_projection_parameters_weight_ = l_self_modules_backbone_modules_patch_embeds_modules_1_modules_projection_parameters_bias_ = (None)
        flatten_3 = x_20.flatten(2)
        x_20 = None
        x_21 = flatten_3.transpose(1, 2)
        flatten_3 = None
        x_22 = torch.nn.functional.layer_norm(
            x_21,
            (192,),
            l_self_modules_backbone_modules_patch_embeds_modules_1_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_patch_embeds_modules_1_modules_norm_parameters_bias_,
            1e-05,
        )
        x_21 = l_self_modules_backbone_modules_patch_embeds_modules_1_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_patch_embeds_modules_1_modules_norm_parameters_bias_ = (None)
        x_23 = torch.nn.functional.dropout(x_22, 0.0, False, False)
        x_22 = None
        layer_norm_8 = torch.nn.functional.layer_norm(
            x_23,
            (192,),
            l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_norm1_parameters_bias_ = (None)
        x_24 = layer_norm_8.view(1, 64, 64, 192)
        layer_norm_8 = None
        x_25 = torch._C._nn.pad(x_24, (0, 0, 0, 6, 0, 6), "constant", None)
        x_24 = None
        mask_2 = torch.zeros((1, 70, 70), device=device(type="cuda", index=0))
        getitem_8 = mask_2[
            (slice(None, None, None), slice(-6, None, None), slice(None, None, None))
        ]
        fill__2 = getitem_8.fill_(1)
        getitem_8 = fill__2 = None
        getitem_9 = mask_2[
            (slice(None, None, None), slice(None, None, None), slice(-6, None, None))
        ]
        fill__3 = getitem_9.fill_(1)
        getitem_9 = fill__3 = None
        reshape_9 = x_25.reshape(1, 10, 7, 10, 7, 192)
        x_25 = None
        x_26 = reshape_9.transpose(2, 3)
        reshape_9 = None
        reshape_10 = mask_2.reshape(1, 10, 7, 10, 7)
        mask_2 = None
        transpose_15 = reshape_10.transpose(2, 3)
        reshape_10 = None
        mask_3 = transpose_15.reshape(1, 100, 49)
        transpose_15 = None
        unsqueeze_3 = mask_3.unsqueeze(2)
        unsqueeze_4 = mask_3.unsqueeze(3)
        mask_3 = None
        attn_mask_2 = unsqueeze_3 - unsqueeze_4
        unsqueeze_3 = unsqueeze_4 = None
        ne_1 = attn_mask_2 != 0
        masked_fill_2 = attn_mask_2.masked_fill(ne_1, -1000.0)
        ne_1 = None
        eq_1 = attn_mask_2 == 0
        attn_mask_2 = None
        attn_mask_3 = masked_fill_2.masked_fill(eq_1, 0.0)
        masked_fill_2 = eq_1 = None
        linear_6 = torch._C._nn.linear(
            x_26,
            l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        x_26 = l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_12 = linear_6.reshape(1, 100, 49, 3, 6, 32)
        linear_6 = None
        qkv_1 = reshape_12.permute(3, 0, 1, 4, 2, 5)
        reshape_12 = None
        q_1 = qkv_1[0]
        k_1 = qkv_1[1]
        v_1 = qkv_1[2]
        qkv_1 = None
        transpose_16 = k_1.transpose(-2, -1)
        k_1 = None
        matmul_2 = q_1 @ transpose_16
        q_1 = transpose_16 = None
        attn_5 = matmul_2 * 0.1767766952966369
        matmul_2 = None
        unsqueeze_5 = attn_mask_3.unsqueeze(2)
        attn_mask_3 = None
        attn_6 = attn_5 + unsqueeze_5
        attn_5 = unsqueeze_5 = None
        attn_7 = attn_6.softmax(dim=-1)
        attn_6 = None
        attn_8 = torch.nn.functional.dropout(attn_7, 0.0, False, False)
        attn_7 = None
        matmul_3 = attn_8 @ v_1
        attn_8 = v_1 = None
        transpose_17 = matmul_3.transpose(2, 3)
        matmul_3 = None
        attn_9 = transpose_17.reshape(1, 10, 10, 7, 7, 192)
        transpose_17 = None
        transpose_18 = attn_9.transpose(2, 3)
        attn_9 = None
        x_27 = transpose_18.reshape(1, 70, 70, 192)
        transpose_18 = None
        getitem_13 = x_27[
            (
                slice(None, None, None),
                slice(None, 64, None),
                slice(None, 64, None),
                slice(None, None, None),
            )
        ]
        x_27 = None
        x_28 = getitem_13.contiguous()
        getitem_13 = None
        x_29 = x_28.reshape(1, 4096, 192)
        x_28 = None
        x_30 = torch._C._nn.linear(
            x_29,
            l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_29 = l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_31 = torch.nn.functional.dropout(x_30, 0.0, False, False)
        x_30 = None
        x_32 = x_23 + x_31
        x_23 = x_31 = None
        layer_norm_9 = torch.nn.functional.layer_norm(
            x_32,
            (192,),
            l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_norm2_parameters_bias_ = (None)
        input_11 = torch._C._nn.linear(
            layer_norm_9,
            l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_9 = l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_12 = torch._C._nn.gelu(input_11, approximate="none")
        input_11 = None
        input_13 = torch.nn.functional.dropout(input_12, 0.0, False, False)
        input_12 = None
        input_14 = torch._C._nn.linear(
            input_13,
            l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_13 = l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_15 = torch.nn.functional.dropout(input_14, 0.0, False, False)
        input_14 = None
        x_33 = x_32 + input_15
        x_32 = input_15 = None
        transpose_19 = x_33.transpose(1, 2)
        x_33 = None
        cnn_feat_1 = transpose_19.view(1, 192, 64, 64)
        transpose_19 = None
        conv2d_4 = torch.conv2d(
            cnn_feat_1,
            l_self_modules_backbone_modules_position_encodings_modules_1_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_position_encodings_modules_1_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            192,
        )
        l_self_modules_backbone_modules_position_encodings_modules_1_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_position_encodings_modules_1_modules_proj_parameters_bias_ = (None)
        x_34 = conv2d_4 + cnn_feat_1
        conv2d_4 = cnn_feat_1 = None
        flatten_4 = x_34.flatten(2)
        x_34 = None
        x_35 = flatten_4.transpose(1, 2)
        flatten_4 = None
        x_q_2 = torch.nn.functional.layer_norm(
            x_35,
            (192,),
            l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_norm1_parameters_bias_ = (None)
        transpose_21 = x_q_2.transpose(1, 2)
        x_kv_5 = transpose_21.reshape(1, 192, 64, 64)
        transpose_21 = None
        x_kv_6 = torch.conv2d(
            x_kv_5,
            l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_sr_parameters_bias_,
            (4, 4),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_5 = l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_5 = x_kv_6.flatten(2)
        x_kv_6 = None
        transpose_22 = flatten_5.transpose(1, 2)
        flatten_5 = None
        x_kv_7 = transpose_22.contiguous()
        transpose_22 = None
        x_kv_8 = torch.nn.functional.layer_norm(
            x_kv_7,
            (192,),
            l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_7 = l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_3 = x_q_2.transpose(0, 1)
        x_q_2 = None
        x_kv_9 = x_kv_8.transpose(0, 1)
        x_kv_8 = None
        multi_head_attention_forward_1 = torch.nn.functional.multi_head_attention_forward(
            x_q_3,
            x_kv_9,
            x_kv_9,
            192,
            6,
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
        x_q_3 = (
            x_kv_9
        ) = l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_1 = multi_head_attention_forward_1[0]
        multi_head_attention_forward_1 = None
        out_1 = attn_output_1.transpose(0, 1)
        attn_output_1 = None
        dropout_13 = torch.nn.functional.dropout(out_1, 0.0, False, False)
        out_1 = None
        add_11 = 0.0 + dropout_13
        dropout_13 = None
        x_36 = x_35 + add_11
        x_35 = add_11 = None
        layer_norm_12 = torch.nn.functional.layer_norm(
            x_36,
            (192,),
            l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_norm2_parameters_bias_ = (None)
        input_16 = torch._C._nn.linear(
            layer_norm_12,
            l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_12 = l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_17 = torch._C._nn.gelu(input_16, approximate="none")
        input_16 = None
        input_18 = torch.nn.functional.dropout(input_17, 0.0, False, False)
        input_17 = None
        input_19 = torch._C._nn.linear(
            input_18,
            l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_18 = l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_20 = torch.nn.functional.dropout(input_19, 0.0, False, False)
        input_19 = None
        x_37 = x_36 + input_20
        x_36 = input_20 = None
        x_38 = torch.nn.functional.layer_norm(
            x_37,
            (192,),
            l_self_modules_backbone_modules_norm_list_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_norm_list_modules_1_parameters_bias_,
            1e-05,
        )
        x_37 = (
            l_self_modules_backbone_modules_norm_list_modules_1_parameters_weight_
        ) = l_self_modules_backbone_modules_norm_list_modules_1_parameters_bias_ = None
        reshape_17 = x_38.reshape(1, 64, 64, -1)
        x_38 = None
        permute_3 = reshape_17.permute(0, 3, 1, 2)
        reshape_17 = None
        x_39 = permute_3.contiguous()
        permute_3 = None
        x_40 = torch.conv2d(
            x_39,
            l_self_modules_backbone_modules_patch_embeds_modules_2_modules_projection_parameters_weight_,
            l_self_modules_backbone_modules_patch_embeds_modules_2_modules_projection_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_patch_embeds_modules_2_modules_projection_parameters_weight_ = l_self_modules_backbone_modules_patch_embeds_modules_2_modules_projection_parameters_bias_ = (None)
        flatten_6 = x_40.flatten(2)
        x_40 = None
        x_41 = flatten_6.transpose(1, 2)
        flatten_6 = None
        x_42 = torch.nn.functional.layer_norm(
            x_41,
            (384,),
            l_self_modules_backbone_modules_patch_embeds_modules_2_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_patch_embeds_modules_2_modules_norm_parameters_bias_,
            1e-05,
        )
        x_41 = l_self_modules_backbone_modules_patch_embeds_modules_2_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_patch_embeds_modules_2_modules_norm_parameters_bias_ = (None)
        x_43 = torch.nn.functional.dropout(x_42, 0.0, False, False)
        x_42 = None
        layer_norm_15 = torch.nn.functional.layer_norm(
            x_43,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm1_parameters_bias_ = (None)
        x_44 = layer_norm_15.view(1, 32, 32, 384)
        layer_norm_15 = None
        x_45 = torch._C._nn.pad(x_44, (0, 0, 0, 3, 0, 3), "constant", None)
        x_44 = None
        mask_4 = torch.zeros((1, 35, 35), device=device(type="cuda", index=0))
        getitem_16 = mask_4[
            (slice(None, None, None), slice(-3, None, None), slice(None, None, None))
        ]
        fill__4 = getitem_16.fill_(1)
        getitem_16 = fill__4 = None
        getitem_17 = mask_4[
            (slice(None, None, None), slice(None, None, None), slice(-3, None, None))
        ]
        fill__5 = getitem_17.fill_(1)
        getitem_17 = fill__5 = None
        reshape_18 = x_45.reshape(1, 5, 7, 5, 7, 384)
        x_45 = None
        x_46 = reshape_18.transpose(2, 3)
        reshape_18 = None
        reshape_19 = mask_4.reshape(1, 5, 7, 5, 7)
        mask_4 = None
        transpose_28 = reshape_19.transpose(2, 3)
        reshape_19 = None
        mask_5 = transpose_28.reshape(1, 25, 49)
        transpose_28 = None
        unsqueeze_6 = mask_5.unsqueeze(2)
        unsqueeze_7 = mask_5.unsqueeze(3)
        mask_5 = None
        attn_mask_4 = unsqueeze_6 - unsqueeze_7
        unsqueeze_6 = unsqueeze_7 = None
        ne_2 = attn_mask_4 != 0
        masked_fill_4 = attn_mask_4.masked_fill(ne_2, -1000.0)
        ne_2 = None
        eq_2 = attn_mask_4 == 0
        attn_mask_4 = None
        attn_mask_5 = masked_fill_4.masked_fill(eq_2, 0.0)
        masked_fill_4 = eq_2 = None
        linear_12 = torch._C._nn.linear(
            x_46,
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        x_46 = l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_21 = linear_12.reshape(1, 25, 49, 3, 12, 32)
        linear_12 = None
        qkv_2 = reshape_21.permute(3, 0, 1, 4, 2, 5)
        reshape_21 = None
        q_2 = qkv_2[0]
        k_2 = qkv_2[1]
        v_2 = qkv_2[2]
        qkv_2 = None
        transpose_29 = k_2.transpose(-2, -1)
        k_2 = None
        matmul_4 = q_2 @ transpose_29
        q_2 = transpose_29 = None
        attn_10 = matmul_4 * 0.1767766952966369
        matmul_4 = None
        unsqueeze_8 = attn_mask_5.unsqueeze(2)
        attn_mask_5 = None
        attn_11 = attn_10 + unsqueeze_8
        attn_10 = unsqueeze_8 = None
        attn_12 = attn_11.softmax(dim=-1)
        attn_11 = None
        attn_13 = torch.nn.functional.dropout(attn_12, 0.0, False, False)
        attn_12 = None
        matmul_5 = attn_13 @ v_2
        attn_13 = v_2 = None
        transpose_30 = matmul_5.transpose(2, 3)
        matmul_5 = None
        attn_14 = transpose_30.reshape(1, 5, 5, 7, 7, 384)
        transpose_30 = None
        transpose_31 = attn_14.transpose(2, 3)
        attn_14 = None
        x_47 = transpose_31.reshape(1, 35, 35, 384)
        transpose_31 = None
        getitem_21 = x_47[
            (
                slice(None, None, None),
                slice(None, 32, None),
                slice(None, 32, None),
                slice(None, None, None),
            )
        ]
        x_47 = None
        x_48 = getitem_21.contiguous()
        getitem_21 = None
        x_49 = x_48.reshape(1, 1024, 384)
        x_48 = None
        x_50 = torch._C._nn.linear(
            x_49,
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_49 = l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_51 = torch.nn.functional.dropout(x_50, 0.0, False, False)
        x_50 = None
        x_52 = x_43 + x_51
        x_43 = x_51 = None
        layer_norm_16 = torch.nn.functional.layer_norm(
            x_52,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_norm2_parameters_bias_ = (None)
        input_21 = torch._C._nn.linear(
            layer_norm_16,
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_16 = l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_22 = torch._C._nn.gelu(input_21, approximate="none")
        input_21 = None
        input_23 = torch.nn.functional.dropout(input_22, 0.0, False, False)
        input_22 = None
        input_24 = torch._C._nn.linear(
            input_23,
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_23 = l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_25 = torch.nn.functional.dropout(input_24, 0.0, False, False)
        input_24 = None
        x_53 = x_52 + input_25
        x_52 = input_25 = None
        transpose_32 = x_53.transpose(1, 2)
        x_53 = None
        cnn_feat_2 = transpose_32.view(1, 384, 32, 32)
        transpose_32 = None
        conv2d_7 = torch.conv2d(
            cnn_feat_2,
            l_self_modules_backbone_modules_position_encodings_modules_2_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_position_encodings_modules_2_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        l_self_modules_backbone_modules_position_encodings_modules_2_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_position_encodings_modules_2_modules_proj_parameters_bias_ = (None)
        x_54 = conv2d_7 + cnn_feat_2
        conv2d_7 = cnn_feat_2 = None
        flatten_7 = x_54.flatten(2)
        x_54 = None
        x_55 = flatten_7.transpose(1, 2)
        flatten_7 = None
        x_q_4 = torch.nn.functional.layer_norm(
            x_55,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_norm1_parameters_bias_ = (None)
        transpose_34 = x_q_4.transpose(1, 2)
        x_kv_10 = transpose_34.reshape(1, 384, 32, 32)
        transpose_34 = None
        x_kv_11 = torch.conv2d(
            x_kv_10,
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_10 = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_8 = x_kv_11.flatten(2)
        x_kv_11 = None
        transpose_35 = flatten_8.transpose(1, 2)
        flatten_8 = None
        x_kv_12 = transpose_35.contiguous()
        transpose_35 = None
        x_kv_13 = torch.nn.functional.layer_norm(
            x_kv_12,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_12 = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_5 = x_q_4.transpose(0, 1)
        x_q_4 = None
        x_kv_14 = x_kv_13.transpose(0, 1)
        x_kv_13 = None
        multi_head_attention_forward_2 = torch.nn.functional.multi_head_attention_forward(
            x_q_5,
            x_kv_14,
            x_kv_14,
            384,
            12,
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
        x_q_5 = (
            x_kv_14
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_2 = multi_head_attention_forward_2[0]
        multi_head_attention_forward_2 = None
        out_2 = attn_output_2.transpose(0, 1)
        attn_output_2 = None
        dropout_21 = torch.nn.functional.dropout(out_2, 0.0, False, False)
        out_2 = None
        add_18 = 0.0 + dropout_21
        dropout_21 = None
        x_56 = x_55 + add_18
        x_55 = add_18 = None
        layer_norm_19 = torch.nn.functional.layer_norm(
            x_56,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_norm2_parameters_bias_ = (None)
        input_26 = torch._C._nn.linear(
            layer_norm_19,
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_19 = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_27 = torch._C._nn.gelu(input_26, approximate="none")
        input_26 = None
        input_28 = torch.nn.functional.dropout(input_27, 0.0, False, False)
        input_27 = None
        input_29 = torch._C._nn.linear(
            input_28,
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_28 = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_30 = torch.nn.functional.dropout(input_29, 0.0, False, False)
        input_29 = None
        x_57 = x_56 + input_30
        x_56 = input_30 = None
        layer_norm_20 = torch.nn.functional.layer_norm(
            x_57,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_norm1_parameters_bias_ = (None)
        x_58 = layer_norm_20.view(1, 32, 32, 384)
        layer_norm_20 = None
        x_59 = torch._C._nn.pad(x_58, (0, 0, 0, 3, 0, 3), "constant", None)
        x_58 = None
        mask_6 = torch.zeros((1, 35, 35), device=device(type="cuda", index=0))
        getitem_24 = mask_6[
            (slice(None, None, None), slice(-3, None, None), slice(None, None, None))
        ]
        fill__6 = getitem_24.fill_(1)
        getitem_24 = fill__6 = None
        getitem_25 = mask_6[
            (slice(None, None, None), slice(None, None, None), slice(-3, None, None))
        ]
        fill__7 = getitem_25.fill_(1)
        getitem_25 = fill__7 = None
        reshape_26 = x_59.reshape(1, 5, 7, 5, 7, 384)
        x_59 = None
        x_60 = reshape_26.transpose(2, 3)
        reshape_26 = None
        reshape_27 = mask_6.reshape(1, 5, 7, 5, 7)
        mask_6 = None
        transpose_40 = reshape_27.transpose(2, 3)
        reshape_27 = None
        mask_7 = transpose_40.reshape(1, 25, 49)
        transpose_40 = None
        unsqueeze_9 = mask_7.unsqueeze(2)
        unsqueeze_10 = mask_7.unsqueeze(3)
        mask_7 = None
        attn_mask_6 = unsqueeze_9 - unsqueeze_10
        unsqueeze_9 = unsqueeze_10 = None
        ne_3 = attn_mask_6 != 0
        masked_fill_6 = attn_mask_6.masked_fill(ne_3, -1000.0)
        ne_3 = None
        eq_3 = attn_mask_6 == 0
        attn_mask_6 = None
        attn_mask_7 = masked_fill_6.masked_fill(eq_3, 0.0)
        masked_fill_6 = eq_3 = None
        linear_18 = torch._C._nn.linear(
            x_60,
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_qkv_parameters_bias_,
        )
        x_60 = l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_29 = linear_18.reshape(1, 25, 49, 3, 12, 32)
        linear_18 = None
        qkv_3 = reshape_29.permute(3, 0, 1, 4, 2, 5)
        reshape_29 = None
        q_3 = qkv_3[0]
        k_3 = qkv_3[1]
        v_3 = qkv_3[2]
        qkv_3 = None
        transpose_41 = k_3.transpose(-2, -1)
        k_3 = None
        matmul_6 = q_3 @ transpose_41
        q_3 = transpose_41 = None
        attn_15 = matmul_6 * 0.1767766952966369
        matmul_6 = None
        unsqueeze_11 = attn_mask_7.unsqueeze(2)
        attn_mask_7 = None
        attn_16 = attn_15 + unsqueeze_11
        attn_15 = unsqueeze_11 = None
        attn_17 = attn_16.softmax(dim=-1)
        attn_16 = None
        attn_18 = torch.nn.functional.dropout(attn_17, 0.0, False, False)
        attn_17 = None
        matmul_7 = attn_18 @ v_3
        attn_18 = v_3 = None
        transpose_42 = matmul_7.transpose(2, 3)
        matmul_7 = None
        attn_19 = transpose_42.reshape(1, 5, 5, 7, 7, 384)
        transpose_42 = None
        transpose_43 = attn_19.transpose(2, 3)
        attn_19 = None
        x_61 = transpose_43.reshape(1, 35, 35, 384)
        transpose_43 = None
        getitem_29 = x_61[
            (
                slice(None, None, None),
                slice(None, 32, None),
                slice(None, 32, None),
                slice(None, None, None),
            )
        ]
        x_61 = None
        x_62 = getitem_29.contiguous()
        getitem_29 = None
        x_63 = x_62.reshape(1, 1024, 384)
        x_62 = None
        x_64 = torch._C._nn.linear(
            x_63,
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_proj_parameters_bias_,
        )
        x_63 = l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_attn_modules_proj_parameters_bias_ = (None)
        x_65 = torch.nn.functional.dropout(x_64, 0.0, False, False)
        x_64 = None
        x_66 = x_57 + x_65
        x_57 = x_65 = None
        layer_norm_21 = torch.nn.functional.layer_norm(
            x_66,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_norm2_parameters_bias_ = (None)
        input_31 = torch._C._nn.linear(
            layer_norm_21,
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_21 = l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_32 = torch._C._nn.gelu(input_31, approximate="none")
        input_31 = None
        input_33 = torch.nn.functional.dropout(input_32, 0.0, False, False)
        input_32 = None
        input_34 = torch._C._nn.linear(
            input_33,
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_33 = l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_35 = torch.nn.functional.dropout(input_34, 0.0, False, False)
        input_34 = None
        x_67 = x_66 + input_35
        x_66 = input_35 = None
        x_q_6 = torch.nn.functional.layer_norm(
            x_67,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_norm1_parameters_bias_ = (None)
        transpose_44 = x_q_6.transpose(1, 2)
        x_kv_15 = transpose_44.reshape(1, 384, 32, 32)
        transpose_44 = None
        x_kv_16 = torch.conv2d(
            x_kv_15,
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_15 = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_9 = x_kv_16.flatten(2)
        x_kv_16 = None
        transpose_45 = flatten_9.transpose(1, 2)
        flatten_9 = None
        x_kv_17 = transpose_45.contiguous()
        transpose_45 = None
        x_kv_18 = torch.nn.functional.layer_norm(
            x_kv_17,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_17 = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_7 = x_q_6.transpose(0, 1)
        x_q_6 = None
        x_kv_19 = x_kv_18.transpose(0, 1)
        x_kv_18 = None
        multi_head_attention_forward_3 = torch.nn.functional.multi_head_attention_forward(
            x_q_7,
            x_kv_19,
            x_kv_19,
            384,
            12,
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
        x_q_7 = (
            x_kv_19
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_3 = multi_head_attention_forward_3[0]
        multi_head_attention_forward_3 = None
        out_3 = attn_output_3.transpose(0, 1)
        attn_output_3 = None
        dropout_28 = torch.nn.functional.dropout(out_3, 0.0, False, False)
        out_3 = None
        add_24 = 0.0 + dropout_28
        dropout_28 = None
        x_68 = x_67 + add_24
        x_67 = add_24 = None
        layer_norm_24 = torch.nn.functional.layer_norm(
            x_68,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_norm2_parameters_bias_ = (None)
        input_36 = torch._C._nn.linear(
            layer_norm_24,
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_24 = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_37 = torch._C._nn.gelu(input_36, approximate="none")
        input_36 = None
        input_38 = torch.nn.functional.dropout(input_37, 0.0, False, False)
        input_37 = None
        input_39 = torch._C._nn.linear(
            input_38,
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_38 = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_3_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_40 = torch.nn.functional.dropout(input_39, 0.0, False, False)
        input_39 = None
        x_69 = x_68 + input_40
        x_68 = input_40 = None
        layer_norm_25 = torch.nn.functional.layer_norm(
            x_69,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_norm1_parameters_bias_ = (None)
        x_70 = layer_norm_25.view(1, 32, 32, 384)
        layer_norm_25 = None
        x_71 = torch._C._nn.pad(x_70, (0, 0, 0, 3, 0, 3), "constant", None)
        x_70 = None
        mask_8 = torch.zeros((1, 35, 35), device=device(type="cuda", index=0))
        getitem_32 = mask_8[
            (slice(None, None, None), slice(-3, None, None), slice(None, None, None))
        ]
        fill__8 = getitem_32.fill_(1)
        getitem_32 = fill__8 = None
        getitem_33 = mask_8[
            (slice(None, None, None), slice(None, None, None), slice(-3, None, None))
        ]
        fill__9 = getitem_33.fill_(1)
        getitem_33 = fill__9 = None
        reshape_34 = x_71.reshape(1, 5, 7, 5, 7, 384)
        x_71 = None
        x_72 = reshape_34.transpose(2, 3)
        reshape_34 = None
        reshape_35 = mask_8.reshape(1, 5, 7, 5, 7)
        mask_8 = None
        transpose_50 = reshape_35.transpose(2, 3)
        reshape_35 = None
        mask_9 = transpose_50.reshape(1, 25, 49)
        transpose_50 = None
        unsqueeze_12 = mask_9.unsqueeze(2)
        unsqueeze_13 = mask_9.unsqueeze(3)
        mask_9 = None
        attn_mask_8 = unsqueeze_12 - unsqueeze_13
        unsqueeze_12 = unsqueeze_13 = None
        ne_4 = attn_mask_8 != 0
        masked_fill_8 = attn_mask_8.masked_fill(ne_4, -1000.0)
        ne_4 = None
        eq_4 = attn_mask_8 == 0
        attn_mask_8 = None
        attn_mask_9 = masked_fill_8.masked_fill(eq_4, 0.0)
        masked_fill_8 = eq_4 = None
        linear_24 = torch._C._nn.linear(
            x_72,
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_qkv_parameters_bias_,
        )
        x_72 = l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_37 = linear_24.reshape(1, 25, 49, 3, 12, 32)
        linear_24 = None
        qkv_4 = reshape_37.permute(3, 0, 1, 4, 2, 5)
        reshape_37 = None
        q_4 = qkv_4[0]
        k_4 = qkv_4[1]
        v_4 = qkv_4[2]
        qkv_4 = None
        transpose_51 = k_4.transpose(-2, -1)
        k_4 = None
        matmul_8 = q_4 @ transpose_51
        q_4 = transpose_51 = None
        attn_20 = matmul_8 * 0.1767766952966369
        matmul_8 = None
        unsqueeze_14 = attn_mask_9.unsqueeze(2)
        attn_mask_9 = None
        attn_21 = attn_20 + unsqueeze_14
        attn_20 = unsqueeze_14 = None
        attn_22 = attn_21.softmax(dim=-1)
        attn_21 = None
        attn_23 = torch.nn.functional.dropout(attn_22, 0.0, False, False)
        attn_22 = None
        matmul_9 = attn_23 @ v_4
        attn_23 = v_4 = None
        transpose_52 = matmul_9.transpose(2, 3)
        matmul_9 = None
        attn_24 = transpose_52.reshape(1, 5, 5, 7, 7, 384)
        transpose_52 = None
        transpose_53 = attn_24.transpose(2, 3)
        attn_24 = None
        x_73 = transpose_53.reshape(1, 35, 35, 384)
        transpose_53 = None
        getitem_37 = x_73[
            (
                slice(None, None, None),
                slice(None, 32, None),
                slice(None, 32, None),
                slice(None, None, None),
            )
        ]
        x_73 = None
        x_74 = getitem_37.contiguous()
        getitem_37 = None
        x_75 = x_74.reshape(1, 1024, 384)
        x_74 = None
        x_76 = torch._C._nn.linear(
            x_75,
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_proj_parameters_bias_,
        )
        x_75 = l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_attn_modules_proj_parameters_bias_ = (None)
        x_77 = torch.nn.functional.dropout(x_76, 0.0, False, False)
        x_76 = None
        x_78 = x_69 + x_77
        x_69 = x_77 = None
        layer_norm_26 = torch.nn.functional.layer_norm(
            x_78,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_norm2_parameters_bias_ = (None)
        input_41 = torch._C._nn.linear(
            layer_norm_26,
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_26 = l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_42 = torch._C._nn.gelu(input_41, approximate="none")
        input_41 = None
        input_43 = torch.nn.functional.dropout(input_42, 0.0, False, False)
        input_42 = None
        input_44 = torch._C._nn.linear(
            input_43,
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_43 = l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_4_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_45 = torch.nn.functional.dropout(input_44, 0.0, False, False)
        input_44 = None
        x_79 = x_78 + input_45
        x_78 = input_45 = None
        x_q_8 = torch.nn.functional.layer_norm(
            x_79,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_norm1_parameters_bias_ = (None)
        transpose_54 = x_q_8.transpose(1, 2)
        x_kv_20 = transpose_54.reshape(1, 384, 32, 32)
        transpose_54 = None
        x_kv_21 = torch.conv2d(
            x_kv_20,
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_20 = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_10 = x_kv_21.flatten(2)
        x_kv_21 = None
        transpose_55 = flatten_10.transpose(1, 2)
        flatten_10 = None
        x_kv_22 = transpose_55.contiguous()
        transpose_55 = None
        x_kv_23 = torch.nn.functional.layer_norm(
            x_kv_22,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_22 = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_9 = x_q_8.transpose(0, 1)
        x_q_8 = None
        x_kv_24 = x_kv_23.transpose(0, 1)
        x_kv_23 = None
        multi_head_attention_forward_4 = torch.nn.functional.multi_head_attention_forward(
            x_q_9,
            x_kv_24,
            x_kv_24,
            384,
            12,
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
        x_q_9 = (
            x_kv_24
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_4 = multi_head_attention_forward_4[0]
        multi_head_attention_forward_4 = None
        out_4 = attn_output_4.transpose(0, 1)
        attn_output_4 = None
        dropout_35 = torch.nn.functional.dropout(out_4, 0.0, False, False)
        out_4 = None
        add_30 = 0.0 + dropout_35
        dropout_35 = None
        x_80 = x_79 + add_30
        x_79 = add_30 = None
        layer_norm_29 = torch.nn.functional.layer_norm(
            x_80,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_norm2_parameters_bias_ = (None)
        input_46 = torch._C._nn.linear(
            layer_norm_29,
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_29 = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_47 = torch._C._nn.gelu(input_46, approximate="none")
        input_46 = None
        input_48 = torch.nn.functional.dropout(input_47, 0.0, False, False)
        input_47 = None
        input_49 = torch._C._nn.linear(
            input_48,
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_48 = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_5_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_50 = torch.nn.functional.dropout(input_49, 0.0, False, False)
        input_49 = None
        x_81 = x_80 + input_50
        x_80 = input_50 = None
        layer_norm_30 = torch.nn.functional.layer_norm(
            x_81,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_norm1_parameters_bias_ = (None)
        x_82 = layer_norm_30.view(1, 32, 32, 384)
        layer_norm_30 = None
        x_83 = torch._C._nn.pad(x_82, (0, 0, 0, 3, 0, 3), "constant", None)
        x_82 = None
        mask_10 = torch.zeros((1, 35, 35), device=device(type="cuda", index=0))
        getitem_40 = mask_10[
            (slice(None, None, None), slice(-3, None, None), slice(None, None, None))
        ]
        fill__10 = getitem_40.fill_(1)
        getitem_40 = fill__10 = None
        getitem_41 = mask_10[
            (slice(None, None, None), slice(None, None, None), slice(-3, None, None))
        ]
        fill__11 = getitem_41.fill_(1)
        getitem_41 = fill__11 = None
        reshape_42 = x_83.reshape(1, 5, 7, 5, 7, 384)
        x_83 = None
        x_84 = reshape_42.transpose(2, 3)
        reshape_42 = None
        reshape_43 = mask_10.reshape(1, 5, 7, 5, 7)
        mask_10 = None
        transpose_60 = reshape_43.transpose(2, 3)
        reshape_43 = None
        mask_11 = transpose_60.reshape(1, 25, 49)
        transpose_60 = None
        unsqueeze_15 = mask_11.unsqueeze(2)
        unsqueeze_16 = mask_11.unsqueeze(3)
        mask_11 = None
        attn_mask_10 = unsqueeze_15 - unsqueeze_16
        unsqueeze_15 = unsqueeze_16 = None
        ne_5 = attn_mask_10 != 0
        masked_fill_10 = attn_mask_10.masked_fill(ne_5, -1000.0)
        ne_5 = None
        eq_5 = attn_mask_10 == 0
        attn_mask_10 = None
        attn_mask_11 = masked_fill_10.masked_fill(eq_5, 0.0)
        masked_fill_10 = eq_5 = None
        linear_30 = torch._C._nn.linear(
            x_84,
            l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_qkv_parameters_bias_,
        )
        x_84 = l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_45 = linear_30.reshape(1, 25, 49, 3, 12, 32)
        linear_30 = None
        qkv_5 = reshape_45.permute(3, 0, 1, 4, 2, 5)
        reshape_45 = None
        q_5 = qkv_5[0]
        k_5 = qkv_5[1]
        v_5 = qkv_5[2]
        qkv_5 = None
        transpose_61 = k_5.transpose(-2, -1)
        k_5 = None
        matmul_10 = q_5 @ transpose_61
        q_5 = transpose_61 = None
        attn_25 = matmul_10 * 0.1767766952966369
        matmul_10 = None
        unsqueeze_17 = attn_mask_11.unsqueeze(2)
        attn_mask_11 = None
        attn_26 = attn_25 + unsqueeze_17
        attn_25 = unsqueeze_17 = None
        attn_27 = attn_26.softmax(dim=-1)
        attn_26 = None
        attn_28 = torch.nn.functional.dropout(attn_27, 0.0, False, False)
        attn_27 = None
        matmul_11 = attn_28 @ v_5
        attn_28 = v_5 = None
        transpose_62 = matmul_11.transpose(2, 3)
        matmul_11 = None
        attn_29 = transpose_62.reshape(1, 5, 5, 7, 7, 384)
        transpose_62 = None
        transpose_63 = attn_29.transpose(2, 3)
        attn_29 = None
        x_85 = transpose_63.reshape(1, 35, 35, 384)
        transpose_63 = None
        getitem_45 = x_85[
            (
                slice(None, None, None),
                slice(None, 32, None),
                slice(None, 32, None),
                slice(None, None, None),
            )
        ]
        x_85 = None
        x_86 = getitem_45.contiguous()
        getitem_45 = None
        x_87 = x_86.reshape(1, 1024, 384)
        x_86 = None
        x_88 = torch._C._nn.linear(
            x_87,
            l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_proj_parameters_bias_,
        )
        x_87 = l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_attn_modules_proj_parameters_bias_ = (None)
        x_89 = torch.nn.functional.dropout(x_88, 0.0, False, False)
        x_88 = None
        x_90 = x_81 + x_89
        x_81 = x_89 = None
        layer_norm_31 = torch.nn.functional.layer_norm(
            x_90,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_norm2_parameters_bias_ = (None)
        input_51 = torch._C._nn.linear(
            layer_norm_31,
            l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_31 = l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_52 = torch._C._nn.gelu(input_51, approximate="none")
        input_51 = None
        input_53 = torch.nn.functional.dropout(input_52, 0.0, False, False)
        input_52 = None
        input_54 = torch._C._nn.linear(
            input_53,
            l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_53 = l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_6_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_55 = torch.nn.functional.dropout(input_54, 0.0, False, False)
        input_54 = None
        x_91 = x_90 + input_55
        x_90 = input_55 = None
        x_q_10 = torch.nn.functional.layer_norm(
            x_91,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_norm1_parameters_bias_ = (None)
        transpose_64 = x_q_10.transpose(1, 2)
        x_kv_25 = transpose_64.reshape(1, 384, 32, 32)
        transpose_64 = None
        x_kv_26 = torch.conv2d(
            x_kv_25,
            l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_25 = l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_11 = x_kv_26.flatten(2)
        x_kv_26 = None
        transpose_65 = flatten_11.transpose(1, 2)
        flatten_11 = None
        x_kv_27 = transpose_65.contiguous()
        transpose_65 = None
        x_kv_28 = torch.nn.functional.layer_norm(
            x_kv_27,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_27 = l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_11 = x_q_10.transpose(0, 1)
        x_q_10 = None
        x_kv_29 = x_kv_28.transpose(0, 1)
        x_kv_28 = None
        multi_head_attention_forward_5 = torch.nn.functional.multi_head_attention_forward(
            x_q_11,
            x_kv_29,
            x_kv_29,
            384,
            12,
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
        x_q_11 = (
            x_kv_29
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_5 = multi_head_attention_forward_5[0]
        multi_head_attention_forward_5 = None
        out_5 = attn_output_5.transpose(0, 1)
        attn_output_5 = None
        dropout_42 = torch.nn.functional.dropout(out_5, 0.0, False, False)
        out_5 = None
        add_36 = 0.0 + dropout_42
        dropout_42 = None
        x_92 = x_91 + add_36
        x_91 = add_36 = None
        layer_norm_34 = torch.nn.functional.layer_norm(
            x_92,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_norm2_parameters_bias_ = (None)
        input_56 = torch._C._nn.linear(
            layer_norm_34,
            l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_34 = l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_57 = torch._C._nn.gelu(input_56, approximate="none")
        input_56 = None
        input_58 = torch.nn.functional.dropout(input_57, 0.0, False, False)
        input_57 = None
        input_59 = torch._C._nn.linear(
            input_58,
            l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_58 = l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_7_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_60 = torch.nn.functional.dropout(input_59, 0.0, False, False)
        input_59 = None
        x_93 = x_92 + input_60
        x_92 = input_60 = None
        layer_norm_35 = torch.nn.functional.layer_norm(
            x_93,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_norm1_parameters_bias_ = (None)
        x_94 = layer_norm_35.view(1, 32, 32, 384)
        layer_norm_35 = None
        x_95 = torch._C._nn.pad(x_94, (0, 0, 0, 3, 0, 3), "constant", None)
        x_94 = None
        mask_12 = torch.zeros((1, 35, 35), device=device(type="cuda", index=0))
        getitem_48 = mask_12[
            (slice(None, None, None), slice(-3, None, None), slice(None, None, None))
        ]
        fill__12 = getitem_48.fill_(1)
        getitem_48 = fill__12 = None
        getitem_49 = mask_12[
            (slice(None, None, None), slice(None, None, None), slice(-3, None, None))
        ]
        fill__13 = getitem_49.fill_(1)
        getitem_49 = fill__13 = None
        reshape_50 = x_95.reshape(1, 5, 7, 5, 7, 384)
        x_95 = None
        x_96 = reshape_50.transpose(2, 3)
        reshape_50 = None
        reshape_51 = mask_12.reshape(1, 5, 7, 5, 7)
        mask_12 = None
        transpose_70 = reshape_51.transpose(2, 3)
        reshape_51 = None
        mask_13 = transpose_70.reshape(1, 25, 49)
        transpose_70 = None
        unsqueeze_18 = mask_13.unsqueeze(2)
        unsqueeze_19 = mask_13.unsqueeze(3)
        mask_13 = None
        attn_mask_12 = unsqueeze_18 - unsqueeze_19
        unsqueeze_18 = unsqueeze_19 = None
        ne_6 = attn_mask_12 != 0
        masked_fill_12 = attn_mask_12.masked_fill(ne_6, -1000.0)
        ne_6 = None
        eq_6 = attn_mask_12 == 0
        attn_mask_12 = None
        attn_mask_13 = masked_fill_12.masked_fill(eq_6, 0.0)
        masked_fill_12 = eq_6 = None
        linear_36 = torch._C._nn.linear(
            x_96,
            l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_qkv_parameters_bias_,
        )
        x_96 = l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_53 = linear_36.reshape(1, 25, 49, 3, 12, 32)
        linear_36 = None
        qkv_6 = reshape_53.permute(3, 0, 1, 4, 2, 5)
        reshape_53 = None
        q_6 = qkv_6[0]
        k_6 = qkv_6[1]
        v_6 = qkv_6[2]
        qkv_6 = None
        transpose_71 = k_6.transpose(-2, -1)
        k_6 = None
        matmul_12 = q_6 @ transpose_71
        q_6 = transpose_71 = None
        attn_30 = matmul_12 * 0.1767766952966369
        matmul_12 = None
        unsqueeze_20 = attn_mask_13.unsqueeze(2)
        attn_mask_13 = None
        attn_31 = attn_30 + unsqueeze_20
        attn_30 = unsqueeze_20 = None
        attn_32 = attn_31.softmax(dim=-1)
        attn_31 = None
        attn_33 = torch.nn.functional.dropout(attn_32, 0.0, False, False)
        attn_32 = None
        matmul_13 = attn_33 @ v_6
        attn_33 = v_6 = None
        transpose_72 = matmul_13.transpose(2, 3)
        matmul_13 = None
        attn_34 = transpose_72.reshape(1, 5, 5, 7, 7, 384)
        transpose_72 = None
        transpose_73 = attn_34.transpose(2, 3)
        attn_34 = None
        x_97 = transpose_73.reshape(1, 35, 35, 384)
        transpose_73 = None
        getitem_53 = x_97[
            (
                slice(None, None, None),
                slice(None, 32, None),
                slice(None, 32, None),
                slice(None, None, None),
            )
        ]
        x_97 = None
        x_98 = getitem_53.contiguous()
        getitem_53 = None
        x_99 = x_98.reshape(1, 1024, 384)
        x_98 = None
        x_100 = torch._C._nn.linear(
            x_99,
            l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_proj_parameters_bias_,
        )
        x_99 = l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_attn_modules_proj_parameters_bias_ = (None)
        x_101 = torch.nn.functional.dropout(x_100, 0.0, False, False)
        x_100 = None
        x_102 = x_93 + x_101
        x_93 = x_101 = None
        layer_norm_36 = torch.nn.functional.layer_norm(
            x_102,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_norm2_parameters_bias_ = (None)
        input_61 = torch._C._nn.linear(
            layer_norm_36,
            l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_36 = l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_62 = torch._C._nn.gelu(input_61, approximate="none")
        input_61 = None
        input_63 = torch.nn.functional.dropout(input_62, 0.0, False, False)
        input_62 = None
        input_64 = torch._C._nn.linear(
            input_63,
            l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_63 = l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_8_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_65 = torch.nn.functional.dropout(input_64, 0.0, False, False)
        input_64 = None
        x_103 = x_102 + input_65
        x_102 = input_65 = None
        x_q_12 = torch.nn.functional.layer_norm(
            x_103,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_norm1_parameters_bias_ = (None)
        transpose_74 = x_q_12.transpose(1, 2)
        x_kv_30 = transpose_74.reshape(1, 384, 32, 32)
        transpose_74 = None
        x_kv_31 = torch.conv2d(
            x_kv_30,
            l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_30 = l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_12 = x_kv_31.flatten(2)
        x_kv_31 = None
        transpose_75 = flatten_12.transpose(1, 2)
        flatten_12 = None
        x_kv_32 = transpose_75.contiguous()
        transpose_75 = None
        x_kv_33 = torch.nn.functional.layer_norm(
            x_kv_32,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_32 = l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_13 = x_q_12.transpose(0, 1)
        x_q_12 = None
        x_kv_34 = x_kv_33.transpose(0, 1)
        x_kv_33 = None
        multi_head_attention_forward_6 = torch.nn.functional.multi_head_attention_forward(
            x_q_13,
            x_kv_34,
            x_kv_34,
            384,
            12,
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
        x_q_13 = (
            x_kv_34
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_6 = multi_head_attention_forward_6[0]
        multi_head_attention_forward_6 = None
        out_6 = attn_output_6.transpose(0, 1)
        attn_output_6 = None
        dropout_49 = torch.nn.functional.dropout(out_6, 0.0, False, False)
        out_6 = None
        add_42 = 0.0 + dropout_49
        dropout_49 = None
        x_104 = x_103 + add_42
        x_103 = add_42 = None
        layer_norm_39 = torch.nn.functional.layer_norm(
            x_104,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_norm2_parameters_bias_ = (None)
        input_66 = torch._C._nn.linear(
            layer_norm_39,
            l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_39 = l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_67 = torch._C._nn.gelu(input_66, approximate="none")
        input_66 = None
        input_68 = torch.nn.functional.dropout(input_67, 0.0, False, False)
        input_67 = None
        input_69 = torch._C._nn.linear(
            input_68,
            l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_68 = l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_9_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_70 = torch.nn.functional.dropout(input_69, 0.0, False, False)
        input_69 = None
        x_105 = x_104 + input_70
        x_104 = input_70 = None
        layer_norm_40 = torch.nn.functional.layer_norm(
            x_105,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_norm1_parameters_bias_ = (None)
        x_106 = layer_norm_40.view(1, 32, 32, 384)
        layer_norm_40 = None
        x_107 = torch._C._nn.pad(x_106, (0, 0, 0, 3, 0, 3), "constant", None)
        x_106 = None
        mask_14 = torch.zeros((1, 35, 35), device=device(type="cuda", index=0))
        getitem_56 = mask_14[
            (slice(None, None, None), slice(-3, None, None), slice(None, None, None))
        ]
        fill__14 = getitem_56.fill_(1)
        getitem_56 = fill__14 = None
        getitem_57 = mask_14[
            (slice(None, None, None), slice(None, None, None), slice(-3, None, None))
        ]
        fill__15 = getitem_57.fill_(1)
        getitem_57 = fill__15 = None
        reshape_58 = x_107.reshape(1, 5, 7, 5, 7, 384)
        x_107 = None
        x_108 = reshape_58.transpose(2, 3)
        reshape_58 = None
        reshape_59 = mask_14.reshape(1, 5, 7, 5, 7)
        mask_14 = None
        transpose_80 = reshape_59.transpose(2, 3)
        reshape_59 = None
        mask_15 = transpose_80.reshape(1, 25, 49)
        transpose_80 = None
        unsqueeze_21 = mask_15.unsqueeze(2)
        unsqueeze_22 = mask_15.unsqueeze(3)
        mask_15 = None
        attn_mask_14 = unsqueeze_21 - unsqueeze_22
        unsqueeze_21 = unsqueeze_22 = None
        ne_7 = attn_mask_14 != 0
        masked_fill_14 = attn_mask_14.masked_fill(ne_7, -1000.0)
        ne_7 = None
        eq_7 = attn_mask_14 == 0
        attn_mask_14 = None
        attn_mask_15 = masked_fill_14.masked_fill(eq_7, 0.0)
        masked_fill_14 = eq_7 = None
        linear_42 = torch._C._nn.linear(
            x_108,
            l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_qkv_parameters_bias_,
        )
        x_108 = l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_61 = linear_42.reshape(1, 25, 49, 3, 12, 32)
        linear_42 = None
        qkv_7 = reshape_61.permute(3, 0, 1, 4, 2, 5)
        reshape_61 = None
        q_7 = qkv_7[0]
        k_7 = qkv_7[1]
        v_7 = qkv_7[2]
        qkv_7 = None
        transpose_81 = k_7.transpose(-2, -1)
        k_7 = None
        matmul_14 = q_7 @ transpose_81
        q_7 = transpose_81 = None
        attn_35 = matmul_14 * 0.1767766952966369
        matmul_14 = None
        unsqueeze_23 = attn_mask_15.unsqueeze(2)
        attn_mask_15 = None
        attn_36 = attn_35 + unsqueeze_23
        attn_35 = unsqueeze_23 = None
        attn_37 = attn_36.softmax(dim=-1)
        attn_36 = None
        attn_38 = torch.nn.functional.dropout(attn_37, 0.0, False, False)
        attn_37 = None
        matmul_15 = attn_38 @ v_7
        attn_38 = v_7 = None
        transpose_82 = matmul_15.transpose(2, 3)
        matmul_15 = None
        attn_39 = transpose_82.reshape(1, 5, 5, 7, 7, 384)
        transpose_82 = None
        transpose_83 = attn_39.transpose(2, 3)
        attn_39 = None
        x_109 = transpose_83.reshape(1, 35, 35, 384)
        transpose_83 = None
        getitem_61 = x_109[
            (
                slice(None, None, None),
                slice(None, 32, None),
                slice(None, 32, None),
                slice(None, None, None),
            )
        ]
        x_109 = None
        x_110 = getitem_61.contiguous()
        getitem_61 = None
        x_111 = x_110.reshape(1, 1024, 384)
        x_110 = None
        x_112 = torch._C._nn.linear(
            x_111,
            l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_proj_parameters_bias_,
        )
        x_111 = l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_attn_modules_proj_parameters_bias_ = (None)
        x_113 = torch.nn.functional.dropout(x_112, 0.0, False, False)
        x_112 = None
        x_114 = x_105 + x_113
        x_105 = x_113 = None
        layer_norm_41 = torch.nn.functional.layer_norm(
            x_114,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_norm2_parameters_bias_ = (None)
        input_71 = torch._C._nn.linear(
            layer_norm_41,
            l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_41 = l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_72 = torch._C._nn.gelu(input_71, approximate="none")
        input_71 = None
        input_73 = torch.nn.functional.dropout(input_72, 0.0, False, False)
        input_72 = None
        input_74 = torch._C._nn.linear(
            input_73,
            l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_73 = l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_10_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_75 = torch.nn.functional.dropout(input_74, 0.0, False, False)
        input_74 = None
        x_115 = x_114 + input_75
        x_114 = input_75 = None
        x_q_14 = torch.nn.functional.layer_norm(
            x_115,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_norm1_parameters_bias_ = (None)
        transpose_84 = x_q_14.transpose(1, 2)
        x_kv_35 = transpose_84.reshape(1, 384, 32, 32)
        transpose_84 = None
        x_kv_36 = torch.conv2d(
            x_kv_35,
            l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_35 = l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_13 = x_kv_36.flatten(2)
        x_kv_36 = None
        transpose_85 = flatten_13.transpose(1, 2)
        flatten_13 = None
        x_kv_37 = transpose_85.contiguous()
        transpose_85 = None
        x_kv_38 = torch.nn.functional.layer_norm(
            x_kv_37,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_37 = l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_15 = x_q_14.transpose(0, 1)
        x_q_14 = None
        x_kv_39 = x_kv_38.transpose(0, 1)
        x_kv_38 = None
        multi_head_attention_forward_7 = torch.nn.functional.multi_head_attention_forward(
            x_q_15,
            x_kv_39,
            x_kv_39,
            384,
            12,
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
        x_q_15 = (
            x_kv_39
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_7 = multi_head_attention_forward_7[0]
        multi_head_attention_forward_7 = None
        out_7 = attn_output_7.transpose(0, 1)
        attn_output_7 = None
        dropout_56 = torch.nn.functional.dropout(out_7, 0.0, False, False)
        out_7 = None
        add_48 = 0.0 + dropout_56
        dropout_56 = None
        x_116 = x_115 + add_48
        x_115 = add_48 = None
        layer_norm_44 = torch.nn.functional.layer_norm(
            x_116,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_norm2_parameters_bias_ = (None)
        input_76 = torch._C._nn.linear(
            layer_norm_44,
            l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_44 = l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_77 = torch._C._nn.gelu(input_76, approximate="none")
        input_76 = None
        input_78 = torch.nn.functional.dropout(input_77, 0.0, False, False)
        input_77 = None
        input_79 = torch._C._nn.linear(
            input_78,
            l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_78 = l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_11_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_80 = torch.nn.functional.dropout(input_79, 0.0, False, False)
        input_79 = None
        x_117 = x_116 + input_80
        x_116 = input_80 = None
        layer_norm_45 = torch.nn.functional.layer_norm(
            x_117,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_norm1_parameters_bias_ = (None)
        x_118 = layer_norm_45.view(1, 32, 32, 384)
        layer_norm_45 = None
        x_119 = torch._C._nn.pad(x_118, (0, 0, 0, 3, 0, 3), "constant", None)
        x_118 = None
        mask_16 = torch.zeros((1, 35, 35), device=device(type="cuda", index=0))
        getitem_64 = mask_16[
            (slice(None, None, None), slice(-3, None, None), slice(None, None, None))
        ]
        fill__16 = getitem_64.fill_(1)
        getitem_64 = fill__16 = None
        getitem_65 = mask_16[
            (slice(None, None, None), slice(None, None, None), slice(-3, None, None))
        ]
        fill__17 = getitem_65.fill_(1)
        getitem_65 = fill__17 = None
        reshape_66 = x_119.reshape(1, 5, 7, 5, 7, 384)
        x_119 = None
        x_120 = reshape_66.transpose(2, 3)
        reshape_66 = None
        reshape_67 = mask_16.reshape(1, 5, 7, 5, 7)
        mask_16 = None
        transpose_90 = reshape_67.transpose(2, 3)
        reshape_67 = None
        mask_17 = transpose_90.reshape(1, 25, 49)
        transpose_90 = None
        unsqueeze_24 = mask_17.unsqueeze(2)
        unsqueeze_25 = mask_17.unsqueeze(3)
        mask_17 = None
        attn_mask_16 = unsqueeze_24 - unsqueeze_25
        unsqueeze_24 = unsqueeze_25 = None
        ne_8 = attn_mask_16 != 0
        masked_fill_16 = attn_mask_16.masked_fill(ne_8, -1000.0)
        ne_8 = None
        eq_8 = attn_mask_16 == 0
        attn_mask_16 = None
        attn_mask_17 = masked_fill_16.masked_fill(eq_8, 0.0)
        masked_fill_16 = eq_8 = None
        linear_48 = torch._C._nn.linear(
            x_120,
            l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_qkv_parameters_bias_,
        )
        x_120 = l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_69 = linear_48.reshape(1, 25, 49, 3, 12, 32)
        linear_48 = None
        qkv_8 = reshape_69.permute(3, 0, 1, 4, 2, 5)
        reshape_69 = None
        q_8 = qkv_8[0]
        k_8 = qkv_8[1]
        v_8 = qkv_8[2]
        qkv_8 = None
        transpose_91 = k_8.transpose(-2, -1)
        k_8 = None
        matmul_16 = q_8 @ transpose_91
        q_8 = transpose_91 = None
        attn_40 = matmul_16 * 0.1767766952966369
        matmul_16 = None
        unsqueeze_26 = attn_mask_17.unsqueeze(2)
        attn_mask_17 = None
        attn_41 = attn_40 + unsqueeze_26
        attn_40 = unsqueeze_26 = None
        attn_42 = attn_41.softmax(dim=-1)
        attn_41 = None
        attn_43 = torch.nn.functional.dropout(attn_42, 0.0, False, False)
        attn_42 = None
        matmul_17 = attn_43 @ v_8
        attn_43 = v_8 = None
        transpose_92 = matmul_17.transpose(2, 3)
        matmul_17 = None
        attn_44 = transpose_92.reshape(1, 5, 5, 7, 7, 384)
        transpose_92 = None
        transpose_93 = attn_44.transpose(2, 3)
        attn_44 = None
        x_121 = transpose_93.reshape(1, 35, 35, 384)
        transpose_93 = None
        getitem_69 = x_121[
            (
                slice(None, None, None),
                slice(None, 32, None),
                slice(None, 32, None),
                slice(None, None, None),
            )
        ]
        x_121 = None
        x_122 = getitem_69.contiguous()
        getitem_69 = None
        x_123 = x_122.reshape(1, 1024, 384)
        x_122 = None
        x_124 = torch._C._nn.linear(
            x_123,
            l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_proj_parameters_bias_,
        )
        x_123 = l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_attn_modules_proj_parameters_bias_ = (None)
        x_125 = torch.nn.functional.dropout(x_124, 0.0, False, False)
        x_124 = None
        x_126 = x_117 + x_125
        x_117 = x_125 = None
        layer_norm_46 = torch.nn.functional.layer_norm(
            x_126,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_norm2_parameters_bias_ = (None)
        input_81 = torch._C._nn.linear(
            layer_norm_46,
            l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_46 = l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_82 = torch._C._nn.gelu(input_81, approximate="none")
        input_81 = None
        input_83 = torch.nn.functional.dropout(input_82, 0.0, False, False)
        input_82 = None
        input_84 = torch._C._nn.linear(
            input_83,
            l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_83 = l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_12_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_85 = torch.nn.functional.dropout(input_84, 0.0, False, False)
        input_84 = None
        x_127 = x_126 + input_85
        x_126 = input_85 = None
        x_q_16 = torch.nn.functional.layer_norm(
            x_127,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_norm1_parameters_bias_ = (None)
        transpose_94 = x_q_16.transpose(1, 2)
        x_kv_40 = transpose_94.reshape(1, 384, 32, 32)
        transpose_94 = None
        x_kv_41 = torch.conv2d(
            x_kv_40,
            l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_40 = l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_14 = x_kv_41.flatten(2)
        x_kv_41 = None
        transpose_95 = flatten_14.transpose(1, 2)
        flatten_14 = None
        x_kv_42 = transpose_95.contiguous()
        transpose_95 = None
        x_kv_43 = torch.nn.functional.layer_norm(
            x_kv_42,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_42 = l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_17 = x_q_16.transpose(0, 1)
        x_q_16 = None
        x_kv_44 = x_kv_43.transpose(0, 1)
        x_kv_43 = None
        multi_head_attention_forward_8 = torch.nn.functional.multi_head_attention_forward(
            x_q_17,
            x_kv_44,
            x_kv_44,
            384,
            12,
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
        x_q_17 = (
            x_kv_44
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_8 = multi_head_attention_forward_8[0]
        multi_head_attention_forward_8 = None
        out_8 = attn_output_8.transpose(0, 1)
        attn_output_8 = None
        dropout_63 = torch.nn.functional.dropout(out_8, 0.0, False, False)
        out_8 = None
        add_54 = 0.0 + dropout_63
        dropout_63 = None
        x_128 = x_127 + add_54
        x_127 = add_54 = None
        layer_norm_49 = torch.nn.functional.layer_norm(
            x_128,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_norm2_parameters_bias_ = (None)
        input_86 = torch._C._nn.linear(
            layer_norm_49,
            l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_49 = l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_87 = torch._C._nn.gelu(input_86, approximate="none")
        input_86 = None
        input_88 = torch.nn.functional.dropout(input_87, 0.0, False, False)
        input_87 = None
        input_89 = torch._C._nn.linear(
            input_88,
            l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_88 = l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_13_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_90 = torch.nn.functional.dropout(input_89, 0.0, False, False)
        input_89 = None
        x_129 = x_128 + input_90
        x_128 = input_90 = None
        layer_norm_50 = torch.nn.functional.layer_norm(
            x_129,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_norm1_parameters_bias_ = (None)
        x_130 = layer_norm_50.view(1, 32, 32, 384)
        layer_norm_50 = None
        x_131 = torch._C._nn.pad(x_130, (0, 0, 0, 3, 0, 3), "constant", None)
        x_130 = None
        mask_18 = torch.zeros((1, 35, 35), device=device(type="cuda", index=0))
        getitem_72 = mask_18[
            (slice(None, None, None), slice(-3, None, None), slice(None, None, None))
        ]
        fill__18 = getitem_72.fill_(1)
        getitem_72 = fill__18 = None
        getitem_73 = mask_18[
            (slice(None, None, None), slice(None, None, None), slice(-3, None, None))
        ]
        fill__19 = getitem_73.fill_(1)
        getitem_73 = fill__19 = None
        reshape_74 = x_131.reshape(1, 5, 7, 5, 7, 384)
        x_131 = None
        x_132 = reshape_74.transpose(2, 3)
        reshape_74 = None
        reshape_75 = mask_18.reshape(1, 5, 7, 5, 7)
        mask_18 = None
        transpose_100 = reshape_75.transpose(2, 3)
        reshape_75 = None
        mask_19 = transpose_100.reshape(1, 25, 49)
        transpose_100 = None
        unsqueeze_27 = mask_19.unsqueeze(2)
        unsqueeze_28 = mask_19.unsqueeze(3)
        mask_19 = None
        attn_mask_18 = unsqueeze_27 - unsqueeze_28
        unsqueeze_27 = unsqueeze_28 = None
        ne_9 = attn_mask_18 != 0
        masked_fill_18 = attn_mask_18.masked_fill(ne_9, -1000.0)
        ne_9 = None
        eq_9 = attn_mask_18 == 0
        attn_mask_18 = None
        attn_mask_19 = masked_fill_18.masked_fill(eq_9, 0.0)
        masked_fill_18 = eq_9 = None
        linear_54 = torch._C._nn.linear(
            x_132,
            l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_qkv_parameters_bias_,
        )
        x_132 = l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_77 = linear_54.reshape(1, 25, 49, 3, 12, 32)
        linear_54 = None
        qkv_9 = reshape_77.permute(3, 0, 1, 4, 2, 5)
        reshape_77 = None
        q_9 = qkv_9[0]
        k_9 = qkv_9[1]
        v_9 = qkv_9[2]
        qkv_9 = None
        transpose_101 = k_9.transpose(-2, -1)
        k_9 = None
        matmul_18 = q_9 @ transpose_101
        q_9 = transpose_101 = None
        attn_45 = matmul_18 * 0.1767766952966369
        matmul_18 = None
        unsqueeze_29 = attn_mask_19.unsqueeze(2)
        attn_mask_19 = None
        attn_46 = attn_45 + unsqueeze_29
        attn_45 = unsqueeze_29 = None
        attn_47 = attn_46.softmax(dim=-1)
        attn_46 = None
        attn_48 = torch.nn.functional.dropout(attn_47, 0.0, False, False)
        attn_47 = None
        matmul_19 = attn_48 @ v_9
        attn_48 = v_9 = None
        transpose_102 = matmul_19.transpose(2, 3)
        matmul_19 = None
        attn_49 = transpose_102.reshape(1, 5, 5, 7, 7, 384)
        transpose_102 = None
        transpose_103 = attn_49.transpose(2, 3)
        attn_49 = None
        x_133 = transpose_103.reshape(1, 35, 35, 384)
        transpose_103 = None
        getitem_77 = x_133[
            (
                slice(None, None, None),
                slice(None, 32, None),
                slice(None, 32, None),
                slice(None, None, None),
            )
        ]
        x_133 = None
        x_134 = getitem_77.contiguous()
        getitem_77 = None
        x_135 = x_134.reshape(1, 1024, 384)
        x_134 = None
        x_136 = torch._C._nn.linear(
            x_135,
            l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_proj_parameters_bias_,
        )
        x_135 = l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_attn_modules_proj_parameters_bias_ = (None)
        x_137 = torch.nn.functional.dropout(x_136, 0.0, False, False)
        x_136 = None
        x_138 = x_129 + x_137
        x_129 = x_137 = None
        layer_norm_51 = torch.nn.functional.layer_norm(
            x_138,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_norm2_parameters_bias_ = (None)
        input_91 = torch._C._nn.linear(
            layer_norm_51,
            l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_51 = l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_92 = torch._C._nn.gelu(input_91, approximate="none")
        input_91 = None
        input_93 = torch.nn.functional.dropout(input_92, 0.0, False, False)
        input_92 = None
        input_94 = torch._C._nn.linear(
            input_93,
            l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_93 = l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_14_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_95 = torch.nn.functional.dropout(input_94, 0.0, False, False)
        input_94 = None
        x_139 = x_138 + input_95
        x_138 = input_95 = None
        x_q_18 = torch.nn.functional.layer_norm(
            x_139,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_norm1_parameters_bias_ = (None)
        transpose_104 = x_q_18.transpose(1, 2)
        x_kv_45 = transpose_104.reshape(1, 384, 32, 32)
        transpose_104 = None
        x_kv_46 = torch.conv2d(
            x_kv_45,
            l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_45 = l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_15 = x_kv_46.flatten(2)
        x_kv_46 = None
        transpose_105 = flatten_15.transpose(1, 2)
        flatten_15 = None
        x_kv_47 = transpose_105.contiguous()
        transpose_105 = None
        x_kv_48 = torch.nn.functional.layer_norm(
            x_kv_47,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_47 = l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_19 = x_q_18.transpose(0, 1)
        x_q_18 = None
        x_kv_49 = x_kv_48.transpose(0, 1)
        x_kv_48 = None
        multi_head_attention_forward_9 = torch.nn.functional.multi_head_attention_forward(
            x_q_19,
            x_kv_49,
            x_kv_49,
            384,
            12,
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
        x_q_19 = (
            x_kv_49
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_9 = multi_head_attention_forward_9[0]
        multi_head_attention_forward_9 = None
        out_9 = attn_output_9.transpose(0, 1)
        attn_output_9 = None
        dropout_70 = torch.nn.functional.dropout(out_9, 0.0, False, False)
        out_9 = None
        add_60 = 0.0 + dropout_70
        dropout_70 = None
        x_140 = x_139 + add_60
        x_139 = add_60 = None
        layer_norm_54 = torch.nn.functional.layer_norm(
            x_140,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_norm2_parameters_bias_ = (None)
        input_96 = torch._C._nn.linear(
            layer_norm_54,
            l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_54 = l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_97 = torch._C._nn.gelu(input_96, approximate="none")
        input_96 = None
        input_98 = torch.nn.functional.dropout(input_97, 0.0, False, False)
        input_97 = None
        input_99 = torch._C._nn.linear(
            input_98,
            l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_98 = l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_15_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_100 = torch.nn.functional.dropout(input_99, 0.0, False, False)
        input_99 = None
        x_141 = x_140 + input_100
        x_140 = input_100 = None
        layer_norm_55 = torch.nn.functional.layer_norm(
            x_141,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_norm1_parameters_bias_ = (None)
        x_142 = layer_norm_55.view(1, 32, 32, 384)
        layer_norm_55 = None
        x_143 = torch._C._nn.pad(x_142, (0, 0, 0, 3, 0, 3), "constant", None)
        x_142 = None
        mask_20 = torch.zeros((1, 35, 35), device=device(type="cuda", index=0))
        getitem_80 = mask_20[
            (slice(None, None, None), slice(-3, None, None), slice(None, None, None))
        ]
        fill__20 = getitem_80.fill_(1)
        getitem_80 = fill__20 = None
        getitem_81 = mask_20[
            (slice(None, None, None), slice(None, None, None), slice(-3, None, None))
        ]
        fill__21 = getitem_81.fill_(1)
        getitem_81 = fill__21 = None
        reshape_82 = x_143.reshape(1, 5, 7, 5, 7, 384)
        x_143 = None
        x_144 = reshape_82.transpose(2, 3)
        reshape_82 = None
        reshape_83 = mask_20.reshape(1, 5, 7, 5, 7)
        mask_20 = None
        transpose_110 = reshape_83.transpose(2, 3)
        reshape_83 = None
        mask_21 = transpose_110.reshape(1, 25, 49)
        transpose_110 = None
        unsqueeze_30 = mask_21.unsqueeze(2)
        unsqueeze_31 = mask_21.unsqueeze(3)
        mask_21 = None
        attn_mask_20 = unsqueeze_30 - unsqueeze_31
        unsqueeze_30 = unsqueeze_31 = None
        ne_10 = attn_mask_20 != 0
        masked_fill_20 = attn_mask_20.masked_fill(ne_10, -1000.0)
        ne_10 = None
        eq_10 = attn_mask_20 == 0
        attn_mask_20 = None
        attn_mask_21 = masked_fill_20.masked_fill(eq_10, 0.0)
        masked_fill_20 = eq_10 = None
        linear_60 = torch._C._nn.linear(
            x_144,
            l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_qkv_parameters_bias_,
        )
        x_144 = l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_85 = linear_60.reshape(1, 25, 49, 3, 12, 32)
        linear_60 = None
        qkv_10 = reshape_85.permute(3, 0, 1, 4, 2, 5)
        reshape_85 = None
        q_10 = qkv_10[0]
        k_10 = qkv_10[1]
        v_10 = qkv_10[2]
        qkv_10 = None
        transpose_111 = k_10.transpose(-2, -1)
        k_10 = None
        matmul_20 = q_10 @ transpose_111
        q_10 = transpose_111 = None
        attn_50 = matmul_20 * 0.1767766952966369
        matmul_20 = None
        unsqueeze_32 = attn_mask_21.unsqueeze(2)
        attn_mask_21 = None
        attn_51 = attn_50 + unsqueeze_32
        attn_50 = unsqueeze_32 = None
        attn_52 = attn_51.softmax(dim=-1)
        attn_51 = None
        attn_53 = torch.nn.functional.dropout(attn_52, 0.0, False, False)
        attn_52 = None
        matmul_21 = attn_53 @ v_10
        attn_53 = v_10 = None
        transpose_112 = matmul_21.transpose(2, 3)
        matmul_21 = None
        attn_54 = transpose_112.reshape(1, 5, 5, 7, 7, 384)
        transpose_112 = None
        transpose_113 = attn_54.transpose(2, 3)
        attn_54 = None
        x_145 = transpose_113.reshape(1, 35, 35, 384)
        transpose_113 = None
        getitem_85 = x_145[
            (
                slice(None, None, None),
                slice(None, 32, None),
                slice(None, 32, None),
                slice(None, None, None),
            )
        ]
        x_145 = None
        x_146 = getitem_85.contiguous()
        getitem_85 = None
        x_147 = x_146.reshape(1, 1024, 384)
        x_146 = None
        x_148 = torch._C._nn.linear(
            x_147,
            l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_proj_parameters_bias_,
        )
        x_147 = l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_attn_modules_proj_parameters_bias_ = (None)
        x_149 = torch.nn.functional.dropout(x_148, 0.0, False, False)
        x_148 = None
        x_150 = x_141 + x_149
        x_141 = x_149 = None
        layer_norm_56 = torch.nn.functional.layer_norm(
            x_150,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_norm2_parameters_bias_ = (None)
        input_101 = torch._C._nn.linear(
            layer_norm_56,
            l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_56 = l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_102 = torch._C._nn.gelu(input_101, approximate="none")
        input_101 = None
        input_103 = torch.nn.functional.dropout(input_102, 0.0, False, False)
        input_102 = None
        input_104 = torch._C._nn.linear(
            input_103,
            l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_103 = l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_16_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_105 = torch.nn.functional.dropout(input_104, 0.0, False, False)
        input_104 = None
        x_151 = x_150 + input_105
        x_150 = input_105 = None
        x_q_20 = torch.nn.functional.layer_norm(
            x_151,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_norm1_parameters_bias_ = (None)
        transpose_114 = x_q_20.transpose(1, 2)
        x_kv_50 = transpose_114.reshape(1, 384, 32, 32)
        transpose_114 = None
        x_kv_51 = torch.conv2d(
            x_kv_50,
            l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_sr_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_kv_50 = l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_sr_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_sr_parameters_bias_ = (None)
        flatten_16 = x_kv_51.flatten(2)
        x_kv_51 = None
        transpose_115 = flatten_16.transpose(1, 2)
        flatten_16 = None
        x_kv_52 = transpose_115.contiguous()
        transpose_115 = None
        x_kv_53 = torch.nn.functional.layer_norm(
            x_kv_52,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_kv_52 = l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_norm_parameters_bias_ = (None)
        x_q_21 = x_q_20.transpose(0, 1)
        x_q_20 = None
        x_kv_54 = x_kv_53.transpose(0, 1)
        x_kv_53 = None
        multi_head_attention_forward_10 = torch.nn.functional.multi_head_attention_forward(
            x_q_21,
            x_kv_54,
            x_kv_54,
            384,
            12,
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
        x_q_21 = (
            x_kv_54
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_10 = multi_head_attention_forward_10[0]
        multi_head_attention_forward_10 = None
        out_10 = attn_output_10.transpose(0, 1)
        attn_output_10 = None
        dropout_77 = torch.nn.functional.dropout(out_10, 0.0, False, False)
        out_10 = None
        add_66 = 0.0 + dropout_77
        dropout_77 = None
        x_152 = x_151 + add_66
        x_151 = add_66 = None
        layer_norm_59 = torch.nn.functional.layer_norm(
            x_152,
            (384,),
            l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_norm2_parameters_bias_ = (None)
        input_106 = torch._C._nn.linear(
            layer_norm_59,
            l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_59 = l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_107 = torch._C._nn.gelu(input_106, approximate="none")
        input_106 = None
        input_108 = torch.nn.functional.dropout(input_107, 0.0, False, False)
        input_107 = None
        input_109 = torch._C._nn.linear(
            input_108,
            l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_108 = l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_17_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_110 = torch.nn.functional.dropout(input_109, 0.0, False, False)
        input_109 = None
        x_153 = x_152 + input_110
        x_152 = input_110 = None
        x_154 = torch.nn.functional.layer_norm(
            x_153,
            (384,),
            l_self_modules_backbone_modules_norm_list_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_norm_list_modules_2_parameters_bias_,
            1e-05,
        )
        x_153 = (
            l_self_modules_backbone_modules_norm_list_modules_2_parameters_weight_
        ) = l_self_modules_backbone_modules_norm_list_modules_2_parameters_bias_ = None
        reshape_90 = x_154.reshape(1, 32, 32, -1)
        x_154 = None
        permute_13 = reshape_90.permute(0, 3, 1, 2)
        reshape_90 = None
        x_155 = permute_13.contiguous()
        permute_13 = None
        x_156 = torch.conv2d(
            x_155,
            l_self_modules_backbone_modules_patch_embeds_modules_3_modules_projection_parameters_weight_,
            l_self_modules_backbone_modules_patch_embeds_modules_3_modules_projection_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_patch_embeds_modules_3_modules_projection_parameters_weight_ = l_self_modules_backbone_modules_patch_embeds_modules_3_modules_projection_parameters_bias_ = (None)
        flatten_17 = x_156.flatten(2)
        x_156 = None
        x_157 = flatten_17.transpose(1, 2)
        flatten_17 = None
        x_158 = torch.nn.functional.layer_norm(
            x_157,
            (768,),
            l_self_modules_backbone_modules_patch_embeds_modules_3_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_patch_embeds_modules_3_modules_norm_parameters_bias_,
            1e-05,
        )
        x_157 = l_self_modules_backbone_modules_patch_embeds_modules_3_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_patch_embeds_modules_3_modules_norm_parameters_bias_ = (None)
        x_159 = torch.nn.functional.dropout(x_158, 0.0, False, False)
        x_158 = None
        layer_norm_62 = torch.nn.functional.layer_norm(
            x_159,
            (768,),
            l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm1_parameters_bias_ = (None)
        x_160 = layer_norm_62.view(1, 16, 16, 768)
        layer_norm_62 = None
        x_161 = torch._C._nn.pad(x_160, (0, 0, 0, 5, 0, 5), "constant", None)
        x_160 = None
        mask_22 = torch.zeros((1, 21, 21), device=device(type="cuda", index=0))
        getitem_88 = mask_22[
            (slice(None, None, None), slice(-5, None, None), slice(None, None, None))
        ]
        fill__22 = getitem_88.fill_(1)
        getitem_88 = fill__22 = None
        getitem_89 = mask_22[
            (slice(None, None, None), slice(None, None, None), slice(-5, None, None))
        ]
        fill__23 = getitem_89.fill_(1)
        getitem_89 = fill__23 = None
        reshape_91 = x_161.reshape(1, 3, 7, 3, 7, 768)
        x_161 = None
        x_162 = reshape_91.transpose(2, 3)
        reshape_91 = None
        reshape_92 = mask_22.reshape(1, 3, 7, 3, 7)
        mask_22 = None
        transpose_121 = reshape_92.transpose(2, 3)
        reshape_92 = None
        mask_23 = transpose_121.reshape(1, 9, 49)
        transpose_121 = None
        unsqueeze_33 = mask_23.unsqueeze(2)
        unsqueeze_34 = mask_23.unsqueeze(3)
        mask_23 = None
        attn_mask_22 = unsqueeze_33 - unsqueeze_34
        unsqueeze_33 = unsqueeze_34 = None
        ne_11 = attn_mask_22 != 0
        masked_fill_22 = attn_mask_22.masked_fill(ne_11, -1000.0)
        ne_11 = None
        eq_11 = attn_mask_22 == 0
        attn_mask_22 = None
        attn_mask_23 = masked_fill_22.masked_fill(eq_11, 0.0)
        masked_fill_22 = eq_11 = None
        linear_66 = torch._C._nn.linear(
            x_162,
            l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        x_162 = l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_94 = linear_66.reshape(1, 9, 49, 3, 24, 32)
        linear_66 = None
        qkv_11 = reshape_94.permute(3, 0, 1, 4, 2, 5)
        reshape_94 = None
        q_11 = qkv_11[0]
        k_11 = qkv_11[1]
        v_11 = qkv_11[2]
        qkv_11 = None
        transpose_122 = k_11.transpose(-2, -1)
        k_11 = None
        matmul_22 = q_11 @ transpose_122
        q_11 = transpose_122 = None
        attn_55 = matmul_22 * 0.1767766952966369
        matmul_22 = None
        unsqueeze_35 = attn_mask_23.unsqueeze(2)
        attn_mask_23 = None
        attn_56 = attn_55 + unsqueeze_35
        attn_55 = unsqueeze_35 = None
        attn_57 = attn_56.softmax(dim=-1)
        attn_56 = None
        attn_58 = torch.nn.functional.dropout(attn_57, 0.0, False, False)
        attn_57 = None
        matmul_23 = attn_58 @ v_11
        attn_58 = v_11 = None
        transpose_123 = matmul_23.transpose(2, 3)
        matmul_23 = None
        attn_59 = transpose_123.reshape(1, 3, 3, 7, 7, 768)
        transpose_123 = None
        transpose_124 = attn_59.transpose(2, 3)
        attn_59 = None
        x_163 = transpose_124.reshape(1, 21, 21, 768)
        transpose_124 = None
        getitem_93 = x_163[
            (
                slice(None, None, None),
                slice(None, 16, None),
                slice(None, 16, None),
                slice(None, None, None),
            )
        ]
        x_163 = None
        x_164 = getitem_93.contiguous()
        getitem_93 = None
        x_165 = x_164.reshape(1, 256, 768)
        x_164 = None
        x_166 = torch._C._nn.linear(
            x_165,
            l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_165 = l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_167 = torch.nn.functional.dropout(x_166, 0.0, False, False)
        x_166 = None
        x_168 = x_159 + x_167
        x_159 = x_167 = None
        layer_norm_63 = torch.nn.functional.layer_norm(
            x_168,
            (768,),
            l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_norm2_parameters_bias_ = (None)
        input_111 = torch._C._nn.linear(
            layer_norm_63,
            l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_63 = l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_112 = torch._C._nn.gelu(input_111, approximate="none")
        input_111 = None
        input_113 = torch.nn.functional.dropout(input_112, 0.0, False, False)
        input_112 = None
        input_114 = torch._C._nn.linear(
            input_113,
            l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_113 = l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_115 = torch.nn.functional.dropout(input_114, 0.0, False, False)
        input_114 = None
        x_169 = x_168 + input_115
        x_168 = input_115 = None
        transpose_125 = x_169.transpose(1, 2)
        x_169 = None
        cnn_feat_3 = transpose_125.view(1, 768, 16, 16)
        transpose_125 = None
        conv2d_18 = torch.conv2d(
            cnn_feat_3,
            l_self_modules_backbone_modules_position_encodings_modules_3_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_position_encodings_modules_3_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        l_self_modules_backbone_modules_position_encodings_modules_3_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_position_encodings_modules_3_modules_proj_parameters_bias_ = (None)
        x_170 = conv2d_18 + cnn_feat_3
        conv2d_18 = cnn_feat_3 = None
        flatten_18 = x_170.flatten(2)
        x_170 = None
        x_171 = flatten_18.transpose(1, 2)
        flatten_18 = None
        x_q_22 = torch.nn.functional.layer_norm(
            x_171,
            (768,),
            l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_norm1_parameters_bias_ = (None)
        x_q_23 = x_q_22.transpose(0, 1)
        x_kv_55 = x_q_22.transpose(0, 1)
        x_q_22 = None
        multi_head_attention_forward_11 = torch.nn.functional.multi_head_attention_forward(
            x_q_23,
            x_kv_55,
            x_kv_55,
            768,
            24,
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
        x_q_23 = (
            x_kv_55
        ) = l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_11 = multi_head_attention_forward_11[0]
        multi_head_attention_forward_11 = None
        out_11 = attn_output_11.transpose(0, 1)
        attn_output_11 = None
        dropout_85 = torch.nn.functional.dropout(out_11, 0.0, False, False)
        out_11 = None
        add_73 = 0.0 + dropout_85
        dropout_85 = None
        x_172 = x_171 + add_73
        x_171 = add_73 = None
        layer_norm_65 = torch.nn.functional.layer_norm(
            x_172,
            (768,),
            l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_norm2_parameters_bias_ = (None)
        input_116 = torch._C._nn.linear(
            layer_norm_65,
            l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_65 = l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_117 = torch._C._nn.gelu(input_116, approximate="none")
        input_116 = None
        input_118 = torch.nn.functional.dropout(input_117, 0.0, False, False)
        input_117 = None
        input_119 = torch._C._nn.linear(
            input_118,
            l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_118 = l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_120 = torch.nn.functional.dropout(input_119, 0.0, False, False)
        input_119 = None
        x_173 = x_172 + input_120
        x_172 = input_120 = None
        x_174 = torch.nn.functional.layer_norm(
            x_173,
            (768,),
            l_self_modules_backbone_modules_norm_list_modules_3_parameters_weight_,
            l_self_modules_backbone_modules_norm_list_modules_3_parameters_bias_,
            1e-05,
        )
        x_173 = (
            l_self_modules_backbone_modules_norm_list_modules_3_parameters_weight_
        ) = l_self_modules_backbone_modules_norm_list_modules_3_parameters_bias_ = None
        reshape_98 = x_174.reshape(1, 16, 16, -1)
        x_174 = None
        permute_15 = reshape_98.permute(0, 3, 1, 2)
        reshape_98 = None
        x_175 = permute_15.contiguous()
        permute_15 = None
        x_176 = torch.conv2d(
            x_19,
            l_self_modules_neck_modules_lateral_convs_modules_0_modules_conv_parameters_weight_,
            l_self_modules_neck_modules_lateral_convs_modules_0_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_19 = l_self_modules_neck_modules_lateral_convs_modules_0_modules_conv_parameters_weight_ = l_self_modules_neck_modules_lateral_convs_modules_0_modules_conv_parameters_bias_ = (None)
        x_177 = torch.conv2d(
            x_39,
            l_self_modules_neck_modules_lateral_convs_modules_1_modules_conv_parameters_weight_,
            l_self_modules_neck_modules_lateral_convs_modules_1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_39 = l_self_modules_neck_modules_lateral_convs_modules_1_modules_conv_parameters_weight_ = l_self_modules_neck_modules_lateral_convs_modules_1_modules_conv_parameters_bias_ = (None)
        x_178 = torch.conv2d(
            x_155,
            l_self_modules_neck_modules_lateral_convs_modules_2_modules_conv_parameters_weight_,
            l_self_modules_neck_modules_lateral_convs_modules_2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_155 = l_self_modules_neck_modules_lateral_convs_modules_2_modules_conv_parameters_weight_ = l_self_modules_neck_modules_lateral_convs_modules_2_modules_conv_parameters_bias_ = (None)
        x_179 = torch.conv2d(
            x_175,
            l_self_modules_neck_modules_lateral_convs_modules_3_modules_conv_parameters_weight_,
            l_self_modules_neck_modules_lateral_convs_modules_3_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_175 = l_self_modules_neck_modules_lateral_convs_modules_3_modules_conv_parameters_weight_ = l_self_modules_neck_modules_lateral_convs_modules_3_modules_conv_parameters_bias_ = (None)
        interpolate = torch.nn.functional.interpolate(
            x_179, (32, 32), None, "nearest", None
        )
        add_76 = x_178 + interpolate
        x_178 = interpolate = None
        interpolate_1 = torch.nn.functional.interpolate(
            add_76, (64, 64), None, "nearest", None
        )
        add_77 = x_177 + interpolate_1
        x_177 = interpolate_1 = None
        interpolate_2 = torch.nn.functional.interpolate(
            add_77, (128, 128), None, "nearest", None
        )
        add_78 = x_176 + interpolate_2
        x_176 = interpolate_2 = None
        x_180 = torch.conv2d(
            add_78,
            l_self_modules_neck_modules_fpn_convs_modules_0_modules_conv_parameters_weight_,
            l_self_modules_neck_modules_fpn_convs_modules_0_modules_conv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        add_78 = l_self_modules_neck_modules_fpn_convs_modules_0_modules_conv_parameters_weight_ = l_self_modules_neck_modules_fpn_convs_modules_0_modules_conv_parameters_bias_ = (None)
        x_181 = torch.conv2d(
            add_77,
            l_self_modules_neck_modules_fpn_convs_modules_1_modules_conv_parameters_weight_,
            l_self_modules_neck_modules_fpn_convs_modules_1_modules_conv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        add_77 = l_self_modules_neck_modules_fpn_convs_modules_1_modules_conv_parameters_weight_ = l_self_modules_neck_modules_fpn_convs_modules_1_modules_conv_parameters_bias_ = (None)
        x_182 = torch.conv2d(
            add_76,
            l_self_modules_neck_modules_fpn_convs_modules_2_modules_conv_parameters_weight_,
            l_self_modules_neck_modules_fpn_convs_modules_2_modules_conv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        add_76 = l_self_modules_neck_modules_fpn_convs_modules_2_modules_conv_parameters_weight_ = l_self_modules_neck_modules_fpn_convs_modules_2_modules_conv_parameters_bias_ = (None)
        x_183 = torch.conv2d(
            x_179,
            l_self_modules_neck_modules_fpn_convs_modules_3_modules_conv_parameters_weight_,
            l_self_modules_neck_modules_fpn_convs_modules_3_modules_conv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_179 = l_self_modules_neck_modules_fpn_convs_modules_3_modules_conv_parameters_weight_ = l_self_modules_neck_modules_fpn_convs_modules_3_modules_conv_parameters_bias_ = (None)
        x_184 = torch.conv2d(
            x_180,
            l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_180 = l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_conv_parameters_weight_ = (None)
        x_185 = torch.nn.functional.batch_norm(
            x_184,
            l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_184 = l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_parameters_bias_ = (None)
        x_186 = torch.nn.functional.relu(x_185, inplace=True)
        x_185 = None
        x_187 = torch.conv2d(
            x_181,
            l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_181 = l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_conv_parameters_weight_ = (None)
        x_188 = torch.nn.functional.batch_norm(
            x_187,
            l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_187 = l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_parameters_bias_ = (None)
        x_189 = torch.nn.functional.relu(x_188, inplace=True)
        x_188 = None
        input_121 = torch.nn.functional.interpolate(
            x_189, [128, 128], None, "bilinear", False
        )
        x_189 = None
        interpolate_4 = torch.nn.functional.interpolate(
            input_121, (128, 128), None, "bilinear", False
        )
        input_121 = None
        output = x_186 + interpolate_4
        x_186 = interpolate_4 = None
        x_190 = torch.conv2d(
            x_182,
            l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_182 = l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_conv_parameters_weight_ = (None)
        x_191 = torch.nn.functional.batch_norm(
            x_190,
            l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_190 = l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_parameters_bias_ = (None)
        x_192 = torch.nn.functional.relu(x_191, inplace=True)
        x_191 = None
        input_122 = torch.nn.functional.interpolate(
            x_192, [64, 64], None, "bilinear", False
        )
        x_192 = None
        x_193 = torch.conv2d(
            input_122,
            l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_122 = l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_conv_parameters_weight_ = (None)
        x_194 = torch.nn.functional.batch_norm(
            x_193,
            l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_193 = l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_parameters_bias_ = (None)
        x_195 = torch.nn.functional.relu(x_194, inplace=True)
        x_194 = None
        input_123 = torch.nn.functional.interpolate(
            x_195, [128, 128], None, "bilinear", False
        )
        x_195 = None
        interpolate_7 = torch.nn.functional.interpolate(
            input_123, (128, 128), None, "bilinear", False
        )
        input_123 = None
        output_1 = output + interpolate_7
        output = interpolate_7 = None
        x_196 = torch.conv2d(
            x_183,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_183 = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_conv_parameters_weight_ = (None)
        x_197 = torch.nn.functional.batch_norm(
            x_196,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_196 = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_parameters_bias_ = (None)
        x_198 = torch.nn.functional.relu(x_197, inplace=True)
        x_197 = None
        input_124 = torch.nn.functional.interpolate(
            x_198, [32, 32], None, "bilinear", False
        )
        x_198 = None
        x_199 = torch.conv2d(
            input_124,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_124 = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_conv_parameters_weight_ = (None)
        x_200 = torch.nn.functional.batch_norm(
            x_199,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_199 = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_parameters_bias_ = (None)
        x_201 = torch.nn.functional.relu(x_200, inplace=True)
        x_200 = None
        input_125 = torch.nn.functional.interpolate(
            x_201, [64, 64], None, "bilinear", False
        )
        x_201 = None
        x_202 = torch.conv2d(
            input_125,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_125 = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_conv_parameters_weight_ = (None)
        x_203 = torch.nn.functional.batch_norm(
            x_202,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_202 = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_parameters_bias_ = (None)
        x_204 = torch.nn.functional.relu(x_203, inplace=True)
        x_203 = None
        input_126 = torch.nn.functional.interpolate(
            x_204, [128, 128], None, "bilinear", False
        )
        x_204 = None
        interpolate_11 = torch.nn.functional.interpolate(
            input_126, (128, 128), None, "bilinear", False
        )
        input_126 = None
        output_2 = output_1 + interpolate_11
        output_1 = interpolate_11 = None
        feat = torch.nn.functional.dropout2d(output_2, 0.1, False, False)
        output_2 = None
        output_3 = torch.conv2d(
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
        return (output_3,)
