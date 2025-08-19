import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_image_encoder_modules_preprocessor_modules_patchifier_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_preprocessor_modules_patchifier_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_pixel_values_: torch.Tensor,
        L_self_modules_image_encoder_modules_preprocessor_modules_patchifier_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_preprocessor_parameters_pos_embed_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_0_modules_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_0_modules_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_0_modules_mlp_modules_fc3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_1_modules_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_1_modules_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_1_modules_mlp_modules_fc3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_2_modules_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_2_modules_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_2_modules_mlp_modules_fc3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_3_modules_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_3_modules_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_3_modules_mlp_modules_fc3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_4_modules_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_4_modules_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_4_modules_mlp_modules_fc3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_5_modules_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_5_modules_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_5_modules_mlp_modules_fc3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_6_modules_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_6_modules_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_6_modules_mlp_modules_fc3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_7_modules_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_7_modules_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_7_modules_mlp_modules_fc3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_8_modules_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_8_modules_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_8_modules_mlp_modules_fc3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_9_modules_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_9_modules_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_9_modules_mlp_modules_fc3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_10_modules_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_10_modules_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_10_modules_mlp_modules_fc3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_11_modules_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_11_modules_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_11_modules_mlp_modules_fc3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_12_modules_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_12_modules_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_12_modules_mlp_modules_fc3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_13_modules_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_13_modules_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_13_modules_mlp_modules_fc3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_14_modules_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_14_modules_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_14_modules_mlp_modules_fc3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_15_modules_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_15_modules_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_15_modules_mlp_modules_fc3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_16_modules_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_16_modules_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_16_modules_mlp_modules_fc3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_17_modules_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_17_modules_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_17_modules_mlp_modules_fc3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_18_modules_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_18_modules_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_18_modules_mlp_modules_fc3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_19_modules_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_19_modules_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_19_modules_mlp_modules_fc3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_20_modules_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_20_modules_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_20_modules_mlp_modules_fc3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_21_modules_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_21_modules_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_21_modules_mlp_modules_fc3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_22_modules_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_22_modules_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_22_modules_mlp_modules_fc3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_23_modules_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_23_modules_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_23_modules_mlp_modules_fc3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_trunk_modules_post_trunk_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_head_parameters_cls_token_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_head_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_head_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_head_modules_linear_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_image_encoder_modules_head_modules_linear_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_image_projector_parameters_weight_: torch.nn.parameter.Parameter,
        L_input_ids_: torch.Tensor,
        L_self_modules_text_encoder_modules_preprocessor_modules_text_embedding_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_preprocessor_parameters_positional_embedding_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_0_modules_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_attention_mask_: torch.Tensor,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_0_modules_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_0_modules_mlp_modules_fc3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_1_modules_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_1_modules_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_1_modules_mlp_modules_fc3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_2_modules_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_2_modules_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_2_modules_mlp_modules_fc3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_3_modules_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_3_modules_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_3_modules_mlp_modules_fc3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_4_modules_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_4_modules_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_4_modules_mlp_modules_fc3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_5_modules_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_5_modules_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_5_modules_mlp_modules_fc3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_6_modules_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_6_modules_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_6_modules_mlp_modules_fc3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_7_modules_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_7_modules_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_7_modules_mlp_modules_fc3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_8_modules_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_8_modules_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_8_modules_mlp_modules_fc3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_9_modules_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_9_modules_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_9_modules_mlp_modules_fc3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_10_modules_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_10_modules_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_10_modules_mlp_modules_fc3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_11_modules_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_11_modules_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_11_modules_mlp_modules_fc3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_encoder_modules_trunk_modules_post_trunk_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_projector_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_parameters_log_logit_scale_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_image_encoder_modules_preprocessor_modules_patchifier_modules_proj_parameters_weight_ = L_self_modules_image_encoder_modules_preprocessor_modules_patchifier_modules_proj_parameters_weight_
        l_self_modules_image_encoder_modules_preprocessor_modules_patchifier_modules_proj_parameters_bias_ = L_self_modules_image_encoder_modules_preprocessor_modules_patchifier_modules_proj_parameters_bias_
        l_pixel_values_ = L_pixel_values_
        l_self_modules_image_encoder_modules_preprocessor_modules_patchifier_modules_norm_parameters_weight_ = L_self_modules_image_encoder_modules_preprocessor_modules_patchifier_modules_norm_parameters_weight_
        l_self_modules_image_encoder_modules_preprocessor_parameters_pos_embed_ = (
            L_self_modules_image_encoder_modules_preprocessor_parameters_pos_embed_
        )
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_0_modules_norm_1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_0_modules_norm_1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_0_modules_norm_2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_0_modules_norm_2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_0_modules_mlp_modules_fc3_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_0_modules_mlp_modules_fc3_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_1_modules_norm_1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_1_modules_norm_1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_1_modules_norm_2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_1_modules_norm_2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_1_modules_mlp_modules_fc3_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_1_modules_mlp_modules_fc3_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_2_modules_norm_1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_2_modules_norm_1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_2_modules_norm_2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_2_modules_norm_2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_2_modules_mlp_modules_fc3_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_2_modules_mlp_modules_fc3_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_3_modules_norm_1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_3_modules_norm_1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_3_modules_norm_2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_3_modules_norm_2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_3_modules_mlp_modules_fc3_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_3_modules_mlp_modules_fc3_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_4_modules_norm_1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_4_modules_norm_1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_4_modules_norm_2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_4_modules_norm_2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_4_modules_mlp_modules_fc3_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_4_modules_mlp_modules_fc3_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_5_modules_norm_1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_5_modules_norm_1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_5_modules_norm_2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_5_modules_norm_2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_5_modules_mlp_modules_fc3_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_5_modules_mlp_modules_fc3_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_6_modules_norm_1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_6_modules_norm_1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_6_modules_norm_2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_6_modules_norm_2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_6_modules_mlp_modules_fc3_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_6_modules_mlp_modules_fc3_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_7_modules_norm_1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_7_modules_norm_1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_7_modules_norm_2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_7_modules_norm_2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_7_modules_mlp_modules_fc3_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_7_modules_mlp_modules_fc3_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_8_modules_norm_1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_8_modules_norm_1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_8_modules_norm_2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_8_modules_norm_2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_8_modules_mlp_modules_fc3_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_8_modules_mlp_modules_fc3_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_9_modules_norm_1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_9_modules_norm_1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_9_modules_norm_2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_9_modules_norm_2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_9_modules_mlp_modules_fc3_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_9_modules_mlp_modules_fc3_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_10_modules_norm_1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_10_modules_norm_1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_10_modules_norm_2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_10_modules_norm_2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_10_modules_mlp_modules_fc3_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_10_modules_mlp_modules_fc3_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_11_modules_norm_1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_11_modules_norm_1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_11_modules_norm_2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_11_modules_norm_2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_11_modules_mlp_modules_fc3_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_11_modules_mlp_modules_fc3_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_12_modules_norm_1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_12_modules_norm_1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_12_modules_norm_2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_12_modules_norm_2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_12_modules_mlp_modules_fc3_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_12_modules_mlp_modules_fc3_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_13_modules_norm_1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_13_modules_norm_1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_13_modules_norm_2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_13_modules_norm_2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_13_modules_mlp_modules_fc3_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_13_modules_mlp_modules_fc3_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_14_modules_norm_1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_14_modules_norm_1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_14_modules_norm_2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_14_modules_norm_2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_14_modules_mlp_modules_fc3_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_14_modules_mlp_modules_fc3_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_15_modules_norm_1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_15_modules_norm_1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_15_modules_norm_2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_15_modules_norm_2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_15_modules_mlp_modules_fc3_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_15_modules_mlp_modules_fc3_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_16_modules_norm_1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_16_modules_norm_1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_16_modules_norm_2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_16_modules_norm_2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_16_modules_mlp_modules_fc3_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_16_modules_mlp_modules_fc3_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_17_modules_norm_1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_17_modules_norm_1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_17_modules_norm_2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_17_modules_norm_2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_17_modules_mlp_modules_fc3_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_17_modules_mlp_modules_fc3_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_18_modules_norm_1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_18_modules_norm_1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_18_modules_norm_2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_18_modules_norm_2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_18_modules_mlp_modules_fc3_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_18_modules_mlp_modules_fc3_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_19_modules_norm_1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_19_modules_norm_1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_19_modules_norm_2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_19_modules_norm_2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_19_modules_mlp_modules_fc3_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_19_modules_mlp_modules_fc3_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_20_modules_norm_1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_20_modules_norm_1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_20_modules_norm_2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_20_modules_norm_2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_20_modules_mlp_modules_fc3_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_20_modules_mlp_modules_fc3_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_21_modules_norm_1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_21_modules_norm_1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_21_modules_norm_2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_21_modules_norm_2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_21_modules_mlp_modules_fc3_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_21_modules_mlp_modules_fc3_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_22_modules_norm_1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_22_modules_norm_1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_22_modules_norm_2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_22_modules_norm_2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_22_modules_mlp_modules_fc3_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_22_modules_mlp_modules_fc3_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_23_modules_norm_1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_23_modules_norm_1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_23_modules_norm_2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_23_modules_norm_2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_23_modules_mlp_modules_fc3_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_23_modules_mlp_modules_fc3_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_image_encoder_modules_trunk_modules_post_trunk_norm_parameters_weight_ = L_self_modules_image_encoder_modules_trunk_modules_post_trunk_norm_parameters_weight_
        l_self_modules_image_encoder_modules_head_parameters_cls_token_ = (
            L_self_modules_image_encoder_modules_head_parameters_cls_token_
        )
        l_self_modules_image_encoder_modules_head_modules_k_parameters_weight_ = (
            L_self_modules_image_encoder_modules_head_modules_k_parameters_weight_
        )
        l_self_modules_image_encoder_modules_head_modules_v_parameters_weight_ = (
            L_self_modules_image_encoder_modules_head_modules_v_parameters_weight_
        )
        l_self_modules_image_encoder_modules_head_modules_linear_parameters_weight_ = (
            L_self_modules_image_encoder_modules_head_modules_linear_parameters_weight_
        )
        l_self_modules_image_encoder_modules_head_modules_linear_parameters_bias_ = (
            L_self_modules_image_encoder_modules_head_modules_linear_parameters_bias_
        )
        l_self_modules_image_projector_parameters_weight_ = (
            L_self_modules_image_projector_parameters_weight_
        )
        l_input_ids_ = L_input_ids_
        l_self_modules_text_encoder_modules_preprocessor_modules_text_embedding_parameters_weight_ = L_self_modules_text_encoder_modules_preprocessor_modules_text_embedding_parameters_weight_
        l_self_modules_text_encoder_modules_preprocessor_parameters_positional_embedding_ = L_self_modules_text_encoder_modules_preprocessor_parameters_positional_embedding_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_0_modules_norm_1_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_0_modules_norm_1_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_attention_mask_ = L_attention_mask_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_0_modules_norm_2_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_0_modules_norm_2_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_0_modules_mlp_modules_fc3_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_0_modules_mlp_modules_fc3_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_1_modules_norm_1_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_1_modules_norm_1_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_1_modules_norm_2_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_1_modules_norm_2_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_1_modules_mlp_modules_fc3_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_1_modules_mlp_modules_fc3_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_2_modules_norm_1_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_2_modules_norm_1_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_2_modules_norm_2_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_2_modules_norm_2_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_2_modules_mlp_modules_fc3_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_2_modules_mlp_modules_fc3_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_3_modules_norm_1_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_3_modules_norm_1_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_3_modules_norm_2_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_3_modules_norm_2_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_3_modules_mlp_modules_fc3_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_3_modules_mlp_modules_fc3_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_4_modules_norm_1_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_4_modules_norm_1_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_4_modules_norm_2_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_4_modules_norm_2_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_4_modules_mlp_modules_fc3_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_4_modules_mlp_modules_fc3_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_5_modules_norm_1_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_5_modules_norm_1_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_5_modules_norm_2_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_5_modules_norm_2_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_5_modules_mlp_modules_fc3_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_5_modules_mlp_modules_fc3_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_6_modules_norm_1_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_6_modules_norm_1_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_6_modules_norm_2_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_6_modules_norm_2_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_6_modules_mlp_modules_fc3_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_6_modules_mlp_modules_fc3_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_7_modules_norm_1_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_7_modules_norm_1_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_7_modules_norm_2_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_7_modules_norm_2_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_7_modules_mlp_modules_fc3_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_7_modules_mlp_modules_fc3_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_8_modules_norm_1_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_8_modules_norm_1_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_8_modules_norm_2_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_8_modules_norm_2_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_8_modules_mlp_modules_fc3_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_8_modules_mlp_modules_fc3_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_9_modules_norm_1_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_9_modules_norm_1_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_9_modules_norm_2_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_9_modules_norm_2_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_9_modules_mlp_modules_fc3_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_9_modules_mlp_modules_fc3_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_10_modules_norm_1_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_10_modules_norm_1_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_10_modules_norm_2_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_10_modules_norm_2_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_10_modules_mlp_modules_fc3_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_10_modules_mlp_modules_fc3_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_11_modules_norm_1_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_11_modules_norm_1_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_11_modules_norm_2_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_11_modules_norm_2_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_11_modules_mlp_modules_fc3_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_11_modules_mlp_modules_fc3_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_text_encoder_modules_trunk_modules_post_trunk_norm_parameters_weight_ = L_self_modules_text_encoder_modules_trunk_modules_post_trunk_norm_parameters_weight_
        l_self_modules_text_projector_parameters_weight_ = (
            L_self_modules_text_projector_parameters_weight_
        )
        l_self_parameters_log_logit_scale_ = L_self_parameters_log_logit_scale_
        conv2d = torch.conv2d(
            l_pixel_values_,
            l_self_modules_image_encoder_modules_preprocessor_modules_patchifier_modules_proj_parameters_weight_,
            l_self_modules_image_encoder_modules_preprocessor_modules_patchifier_modules_proj_parameters_bias_,
            (14, 14),
            (0, 0),
            (1, 1),
            1,
        )
        l_pixel_values_ = l_self_modules_image_encoder_modules_preprocessor_modules_patchifier_modules_proj_parameters_weight_ = l_self_modules_image_encoder_modules_preprocessor_modules_patchifier_modules_proj_parameters_bias_ = (None)
        flatten = conv2d.flatten(2)
        conv2d = None
        x = flatten.transpose(1, 2)
        flatten = None
        float_1 = x.float()
        pow_1 = float_1.pow(2)
        mean = pow_1.mean(-1, keepdim=True)
        pow_1 = None
        add = mean + 1e-05
        mean = None
        rsqrt = torch.rsqrt(add)
        add = None
        mul = float_1 * rsqrt
        float_1 = rsqrt = None
        output = mul.type_as(x)
        mul = x = None
        x_1 = (
            output
            * l_self_modules_image_encoder_modules_preprocessor_modules_patchifier_modules_norm_parameters_weight_
        )
        output = l_self_modules_image_encoder_modules_preprocessor_modules_patchifier_modules_norm_parameters_weight_ = (None)
        pos_embed = (
            l_self_modules_image_encoder_modules_preprocessor_parameters_pos_embed_.to(
                device(type="cuda", index=0)
            )
        )
        l_self_modules_image_encoder_modules_preprocessor_parameters_pos_embed_ = None
        getitem = pos_embed[(slice(None, None, None), slice(None, 256, None))]
        pos_embed = None
        tokens = x_1 + getitem
        x_1 = getitem = None
        float_2 = tokens.float()
        pow_2 = float_2.pow(2)
        mean_1 = pow_2.mean(-1, keepdim=True)
        pow_2 = None
        add_2 = mean_1 + 1e-05
        mean_1 = None
        rsqrt_1 = torch.rsqrt(add_2)
        add_2 = None
        mul_2 = float_2 * rsqrt_1
        float_2 = rsqrt_1 = None
        output_1 = mul_2.type_as(tokens)
        mul_2 = None
        mul_3 = (
            output_1
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_0_modules_norm_1_parameters_weight_
        )
        output_1 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_0_modules_norm_1_parameters_weight_ = (None)
        linear = torch._C._nn.linear(
            mul_3,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_,
            None,
        )
        mul_3 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_ = (None)
        reshape = linear.reshape(1, 256, 3, 8, 128)
        linear = None
        qkv = reshape.permute(2, 0, 3, 1, 4)
        reshape = None
        unbind = qkv.unbind(0)
        qkv = None
        q = unbind[0]
        k = unbind[1]
        v = unbind[2]
        unbind = None
        x_2 = torch._C._nn.scaled_dot_product_attention(q, k, v, is_causal=False)
        q = k = v = None
        transpose_1 = x_2.transpose(1, 2)
        x_2 = None
        contiguous = transpose_1.contiguous()
        transpose_1 = None
        x_3 = contiguous.reshape(1, 256, 1024)
        contiguous = None
        x_4 = torch._C._nn.linear(
            x_3,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_,
            None,
        )
        x_3 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_ = (None)
        x_5 = torch.nn.functional.dropout(x_4, 0.0, False, False)
        x_4 = None
        x_6 = tokens + x_5
        tokens = x_5 = None
        float_3 = x_6.float()
        pow_3 = float_3.pow(2)
        mean_2 = pow_3.mean(-1, keepdim=True)
        pow_3 = None
        add_4 = mean_2 + 1e-05
        mean_2 = None
        rsqrt_2 = torch.rsqrt(add_4)
        add_4 = None
        mul_4 = float_3 * rsqrt_2
        float_3 = rsqrt_2 = None
        output_2 = mul_4.type_as(x_6)
        mul_4 = None
        mul_5 = (
            output_2
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_0_modules_norm_2_parameters_weight_
        )
        output_2 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_0_modules_norm_2_parameters_weight_ = (None)
        linear_2 = torch._C._nn.linear(
            mul_5,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = (
            None
        )
        silu = torch.nn.functional.silu(linear_2)
        linear_2 = None
        linear_3 = torch._C._nn.linear(
            mul_5,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_0_modules_mlp_modules_fc3_parameters_weight_,
            None,
        )
        mul_5 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_0_modules_mlp_modules_fc3_parameters_weight_ = (None)
        x_7 = silu * linear_3
        silu = linear_3 = None
        x_8 = torch._C._nn.linear(
            x_7,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_7 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_9 = x_6 + x_8
        x_6 = x_8 = None
        float_4 = x_9.float()
        pow_4 = float_4.pow(2)
        mean_3 = pow_4.mean(-1, keepdim=True)
        pow_4 = None
        add_6 = mean_3 + 1e-05
        mean_3 = None
        rsqrt_3 = torch.rsqrt(add_6)
        add_6 = None
        mul_7 = float_4 * rsqrt_3
        float_4 = rsqrt_3 = None
        output_3 = mul_7.type_as(x_9)
        mul_7 = None
        mul_8 = (
            output_3
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_1_modules_norm_1_parameters_weight_
        )
        output_3 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_1_modules_norm_1_parameters_weight_ = (None)
        linear_5 = torch._C._nn.linear(
            mul_8,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_,
            None,
        )
        mul_8 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_ = (None)
        reshape_2 = linear_5.reshape(1, 256, 3, 8, 128)
        linear_5 = None
        qkv_1 = reshape_2.permute(2, 0, 3, 1, 4)
        reshape_2 = None
        unbind_1 = qkv_1.unbind(0)
        qkv_1 = None
        q_1 = unbind_1[0]
        k_1 = unbind_1[1]
        v_1 = unbind_1[2]
        unbind_1 = None
        x_10 = torch._C._nn.scaled_dot_product_attention(q_1, k_1, v_1, is_causal=False)
        q_1 = k_1 = v_1 = None
        transpose_2 = x_10.transpose(1, 2)
        x_10 = None
        contiguous_1 = transpose_2.contiguous()
        transpose_2 = None
        x_11 = contiguous_1.reshape(1, 256, 1024)
        contiguous_1 = None
        x_12 = torch._C._nn.linear(
            x_11,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_,
            None,
        )
        x_11 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_ = (None)
        x_13 = torch.nn.functional.dropout(x_12, 0.0, False, False)
        x_12 = None
        x_14 = x_9 + x_13
        x_9 = x_13 = None
        float_5 = x_14.float()
        pow_5 = float_5.pow(2)
        mean_4 = pow_5.mean(-1, keepdim=True)
        pow_5 = None
        add_8 = mean_4 + 1e-05
        mean_4 = None
        rsqrt_4 = torch.rsqrt(add_8)
        add_8 = None
        mul_9 = float_5 * rsqrt_4
        float_5 = rsqrt_4 = None
        output_4 = mul_9.type_as(x_14)
        mul_9 = None
        mul_10 = (
            output_4
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_1_modules_norm_2_parameters_weight_
        )
        output_4 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_1_modules_norm_2_parameters_weight_ = (None)
        linear_7 = torch._C._nn.linear(
            mul_10,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = (
            None
        )
        silu_1 = torch.nn.functional.silu(linear_7)
        linear_7 = None
        linear_8 = torch._C._nn.linear(
            mul_10,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_1_modules_mlp_modules_fc3_parameters_weight_,
            None,
        )
        mul_10 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_1_modules_mlp_modules_fc3_parameters_weight_ = (None)
        x_15 = silu_1 * linear_8
        silu_1 = linear_8 = None
        x_16 = torch._C._nn.linear(
            x_15,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_15 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_17 = x_14 + x_16
        x_14 = x_16 = None
        float_6 = x_17.float()
        pow_6 = float_6.pow(2)
        mean_5 = pow_6.mean(-1, keepdim=True)
        pow_6 = None
        add_10 = mean_5 + 1e-05
        mean_5 = None
        rsqrt_5 = torch.rsqrt(add_10)
        add_10 = None
        mul_12 = float_6 * rsqrt_5
        float_6 = rsqrt_5 = None
        output_5 = mul_12.type_as(x_17)
        mul_12 = None
        mul_13 = (
            output_5
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_2_modules_norm_1_parameters_weight_
        )
        output_5 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_2_modules_norm_1_parameters_weight_ = (None)
        linear_10 = torch._C._nn.linear(
            mul_13,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_,
            None,
        )
        mul_13 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_ = (None)
        reshape_4 = linear_10.reshape(1, 256, 3, 8, 128)
        linear_10 = None
        qkv_2 = reshape_4.permute(2, 0, 3, 1, 4)
        reshape_4 = None
        unbind_2 = qkv_2.unbind(0)
        qkv_2 = None
        q_2 = unbind_2[0]
        k_2 = unbind_2[1]
        v_2 = unbind_2[2]
        unbind_2 = None
        x_18 = torch._C._nn.scaled_dot_product_attention(q_2, k_2, v_2, is_causal=False)
        q_2 = k_2 = v_2 = None
        transpose_3 = x_18.transpose(1, 2)
        x_18 = None
        contiguous_2 = transpose_3.contiguous()
        transpose_3 = None
        x_19 = contiguous_2.reshape(1, 256, 1024)
        contiguous_2 = None
        x_20 = torch._C._nn.linear(
            x_19,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_,
            None,
        )
        x_19 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_ = (None)
        x_21 = torch.nn.functional.dropout(x_20, 0.0, False, False)
        x_20 = None
        x_22 = x_17 + x_21
        x_17 = x_21 = None
        float_7 = x_22.float()
        pow_7 = float_7.pow(2)
        mean_6 = pow_7.mean(-1, keepdim=True)
        pow_7 = None
        add_12 = mean_6 + 1e-05
        mean_6 = None
        rsqrt_6 = torch.rsqrt(add_12)
        add_12 = None
        mul_14 = float_7 * rsqrt_6
        float_7 = rsqrt_6 = None
        output_6 = mul_14.type_as(x_22)
        mul_14 = None
        mul_15 = (
            output_6
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_2_modules_norm_2_parameters_weight_
        )
        output_6 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_2_modules_norm_2_parameters_weight_ = (None)
        linear_12 = torch._C._nn.linear(
            mul_15,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = (
            None
        )
        silu_2 = torch.nn.functional.silu(linear_12)
        linear_12 = None
        linear_13 = torch._C._nn.linear(
            mul_15,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_2_modules_mlp_modules_fc3_parameters_weight_,
            None,
        )
        mul_15 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_2_modules_mlp_modules_fc3_parameters_weight_ = (None)
        x_23 = silu_2 * linear_13
        silu_2 = linear_13 = None
        x_24 = torch._C._nn.linear(
            x_23,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_23 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_25 = x_22 + x_24
        x_22 = x_24 = None
        float_8 = x_25.float()
        pow_8 = float_8.pow(2)
        mean_7 = pow_8.mean(-1, keepdim=True)
        pow_8 = None
        add_14 = mean_7 + 1e-05
        mean_7 = None
        rsqrt_7 = torch.rsqrt(add_14)
        add_14 = None
        mul_17 = float_8 * rsqrt_7
        float_8 = rsqrt_7 = None
        output_7 = mul_17.type_as(x_25)
        mul_17 = None
        mul_18 = (
            output_7
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_3_modules_norm_1_parameters_weight_
        )
        output_7 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_3_modules_norm_1_parameters_weight_ = (None)
        linear_15 = torch._C._nn.linear(
            mul_18,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_,
            None,
        )
        mul_18 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_ = (None)
        reshape_6 = linear_15.reshape(1, 256, 3, 8, 128)
        linear_15 = None
        qkv_3 = reshape_6.permute(2, 0, 3, 1, 4)
        reshape_6 = None
        unbind_3 = qkv_3.unbind(0)
        qkv_3 = None
        q_3 = unbind_3[0]
        k_3 = unbind_3[1]
        v_3 = unbind_3[2]
        unbind_3 = None
        x_26 = torch._C._nn.scaled_dot_product_attention(q_3, k_3, v_3, is_causal=False)
        q_3 = k_3 = v_3 = None
        transpose_4 = x_26.transpose(1, 2)
        x_26 = None
        contiguous_3 = transpose_4.contiguous()
        transpose_4 = None
        x_27 = contiguous_3.reshape(1, 256, 1024)
        contiguous_3 = None
        x_28 = torch._C._nn.linear(
            x_27,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_,
            None,
        )
        x_27 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_ = (None)
        x_29 = torch.nn.functional.dropout(x_28, 0.0, False, False)
        x_28 = None
        x_30 = x_25 + x_29
        x_25 = x_29 = None
        float_9 = x_30.float()
        pow_9 = float_9.pow(2)
        mean_8 = pow_9.mean(-1, keepdim=True)
        pow_9 = None
        add_16 = mean_8 + 1e-05
        mean_8 = None
        rsqrt_8 = torch.rsqrt(add_16)
        add_16 = None
        mul_19 = float_9 * rsqrt_8
        float_9 = rsqrt_8 = None
        output_8 = mul_19.type_as(x_30)
        mul_19 = None
        mul_20 = (
            output_8
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_3_modules_norm_2_parameters_weight_
        )
        output_8 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_3_modules_norm_2_parameters_weight_ = (None)
        linear_17 = torch._C._nn.linear(
            mul_20,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = (
            None
        )
        silu_3 = torch.nn.functional.silu(linear_17)
        linear_17 = None
        linear_18 = torch._C._nn.linear(
            mul_20,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_3_modules_mlp_modules_fc3_parameters_weight_,
            None,
        )
        mul_20 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_3_modules_mlp_modules_fc3_parameters_weight_ = (None)
        x_31 = silu_3 * linear_18
        silu_3 = linear_18 = None
        x_32 = torch._C._nn.linear(
            x_31,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_31 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_33 = x_30 + x_32
        x_30 = x_32 = None
        float_10 = x_33.float()
        pow_10 = float_10.pow(2)
        mean_9 = pow_10.mean(-1, keepdim=True)
        pow_10 = None
        add_18 = mean_9 + 1e-05
        mean_9 = None
        rsqrt_9 = torch.rsqrt(add_18)
        add_18 = None
        mul_22 = float_10 * rsqrt_9
        float_10 = rsqrt_9 = None
        output_9 = mul_22.type_as(x_33)
        mul_22 = None
        mul_23 = (
            output_9
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_4_modules_norm_1_parameters_weight_
        )
        output_9 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_4_modules_norm_1_parameters_weight_ = (None)
        linear_20 = torch._C._nn.linear(
            mul_23,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_,
            None,
        )
        mul_23 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_ = (None)
        reshape_8 = linear_20.reshape(1, 256, 3, 8, 128)
        linear_20 = None
        qkv_4 = reshape_8.permute(2, 0, 3, 1, 4)
        reshape_8 = None
        unbind_4 = qkv_4.unbind(0)
        qkv_4 = None
        q_4 = unbind_4[0]
        k_4 = unbind_4[1]
        v_4 = unbind_4[2]
        unbind_4 = None
        x_34 = torch._C._nn.scaled_dot_product_attention(q_4, k_4, v_4, is_causal=False)
        q_4 = k_4 = v_4 = None
        transpose_5 = x_34.transpose(1, 2)
        x_34 = None
        contiguous_4 = transpose_5.contiguous()
        transpose_5 = None
        x_35 = contiguous_4.reshape(1, 256, 1024)
        contiguous_4 = None
        x_36 = torch._C._nn.linear(
            x_35,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_,
            None,
        )
        x_35 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_ = (None)
        x_37 = torch.nn.functional.dropout(x_36, 0.0, False, False)
        x_36 = None
        x_38 = x_33 + x_37
        x_33 = x_37 = None
        float_11 = x_38.float()
        pow_11 = float_11.pow(2)
        mean_10 = pow_11.mean(-1, keepdim=True)
        pow_11 = None
        add_20 = mean_10 + 1e-05
        mean_10 = None
        rsqrt_10 = torch.rsqrt(add_20)
        add_20 = None
        mul_24 = float_11 * rsqrt_10
        float_11 = rsqrt_10 = None
        output_10 = mul_24.type_as(x_38)
        mul_24 = None
        mul_25 = (
            output_10
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_4_modules_norm_2_parameters_weight_
        )
        output_10 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_4_modules_norm_2_parameters_weight_ = (None)
        linear_22 = torch._C._nn.linear(
            mul_25,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = (
            None
        )
        silu_4 = torch.nn.functional.silu(linear_22)
        linear_22 = None
        linear_23 = torch._C._nn.linear(
            mul_25,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_4_modules_mlp_modules_fc3_parameters_weight_,
            None,
        )
        mul_25 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_4_modules_mlp_modules_fc3_parameters_weight_ = (None)
        x_39 = silu_4 * linear_23
        silu_4 = linear_23 = None
        x_40 = torch._C._nn.linear(
            x_39,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_39 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_41 = x_38 + x_40
        x_38 = x_40 = None
        float_12 = x_41.float()
        pow_12 = float_12.pow(2)
        mean_11 = pow_12.mean(-1, keepdim=True)
        pow_12 = None
        add_22 = mean_11 + 1e-05
        mean_11 = None
        rsqrt_11 = torch.rsqrt(add_22)
        add_22 = None
        mul_27 = float_12 * rsqrt_11
        float_12 = rsqrt_11 = None
        output_11 = mul_27.type_as(x_41)
        mul_27 = None
        mul_28 = (
            output_11
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_5_modules_norm_1_parameters_weight_
        )
        output_11 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_5_modules_norm_1_parameters_weight_ = (None)
        linear_25 = torch._C._nn.linear(
            mul_28,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_,
            None,
        )
        mul_28 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_ = (None)
        reshape_10 = linear_25.reshape(1, 256, 3, 8, 128)
        linear_25 = None
        qkv_5 = reshape_10.permute(2, 0, 3, 1, 4)
        reshape_10 = None
        unbind_5 = qkv_5.unbind(0)
        qkv_5 = None
        q_5 = unbind_5[0]
        k_5 = unbind_5[1]
        v_5 = unbind_5[2]
        unbind_5 = None
        x_42 = torch._C._nn.scaled_dot_product_attention(q_5, k_5, v_5, is_causal=False)
        q_5 = k_5 = v_5 = None
        transpose_6 = x_42.transpose(1, 2)
        x_42 = None
        contiguous_5 = transpose_6.contiguous()
        transpose_6 = None
        x_43 = contiguous_5.reshape(1, 256, 1024)
        contiguous_5 = None
        x_44 = torch._C._nn.linear(
            x_43,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_,
            None,
        )
        x_43 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_ = (None)
        x_45 = torch.nn.functional.dropout(x_44, 0.0, False, False)
        x_44 = None
        x_46 = x_41 + x_45
        x_41 = x_45 = None
        float_13 = x_46.float()
        pow_13 = float_13.pow(2)
        mean_12 = pow_13.mean(-1, keepdim=True)
        pow_13 = None
        add_24 = mean_12 + 1e-05
        mean_12 = None
        rsqrt_12 = torch.rsqrt(add_24)
        add_24 = None
        mul_29 = float_13 * rsqrt_12
        float_13 = rsqrt_12 = None
        output_12 = mul_29.type_as(x_46)
        mul_29 = None
        mul_30 = (
            output_12
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_5_modules_norm_2_parameters_weight_
        )
        output_12 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_5_modules_norm_2_parameters_weight_ = (None)
        linear_27 = torch._C._nn.linear(
            mul_30,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = (
            None
        )
        silu_5 = torch.nn.functional.silu(linear_27)
        linear_27 = None
        linear_28 = torch._C._nn.linear(
            mul_30,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_5_modules_mlp_modules_fc3_parameters_weight_,
            None,
        )
        mul_30 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_5_modules_mlp_modules_fc3_parameters_weight_ = (None)
        x_47 = silu_5 * linear_28
        silu_5 = linear_28 = None
        x_48 = torch._C._nn.linear(
            x_47,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_47 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_49 = x_46 + x_48
        x_46 = x_48 = None
        float_14 = x_49.float()
        pow_14 = float_14.pow(2)
        mean_13 = pow_14.mean(-1, keepdim=True)
        pow_14 = None
        add_26 = mean_13 + 1e-05
        mean_13 = None
        rsqrt_13 = torch.rsqrt(add_26)
        add_26 = None
        mul_32 = float_14 * rsqrt_13
        float_14 = rsqrt_13 = None
        output_13 = mul_32.type_as(x_49)
        mul_32 = None
        mul_33 = (
            output_13
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_6_modules_norm_1_parameters_weight_
        )
        output_13 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_6_modules_norm_1_parameters_weight_ = (None)
        linear_30 = torch._C._nn.linear(
            mul_33,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_,
            None,
        )
        mul_33 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_ = (None)
        reshape_12 = linear_30.reshape(1, 256, 3, 8, 128)
        linear_30 = None
        qkv_6 = reshape_12.permute(2, 0, 3, 1, 4)
        reshape_12 = None
        unbind_6 = qkv_6.unbind(0)
        qkv_6 = None
        q_6 = unbind_6[0]
        k_6 = unbind_6[1]
        v_6 = unbind_6[2]
        unbind_6 = None
        x_50 = torch._C._nn.scaled_dot_product_attention(q_6, k_6, v_6, is_causal=False)
        q_6 = k_6 = v_6 = None
        transpose_7 = x_50.transpose(1, 2)
        x_50 = None
        contiguous_6 = transpose_7.contiguous()
        transpose_7 = None
        x_51 = contiguous_6.reshape(1, 256, 1024)
        contiguous_6 = None
        x_52 = torch._C._nn.linear(
            x_51,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_,
            None,
        )
        x_51 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_ = (None)
        x_53 = torch.nn.functional.dropout(x_52, 0.0, False, False)
        x_52 = None
        x_54 = x_49 + x_53
        x_49 = x_53 = None
        float_15 = x_54.float()
        pow_15 = float_15.pow(2)
        mean_14 = pow_15.mean(-1, keepdim=True)
        pow_15 = None
        add_28 = mean_14 + 1e-05
        mean_14 = None
        rsqrt_14 = torch.rsqrt(add_28)
        add_28 = None
        mul_34 = float_15 * rsqrt_14
        float_15 = rsqrt_14 = None
        output_14 = mul_34.type_as(x_54)
        mul_34 = None
        mul_35 = (
            output_14
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_6_modules_norm_2_parameters_weight_
        )
        output_14 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_6_modules_norm_2_parameters_weight_ = (None)
        linear_32 = torch._C._nn.linear(
            mul_35,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = (
            None
        )
        silu_6 = torch.nn.functional.silu(linear_32)
        linear_32 = None
        linear_33 = torch._C._nn.linear(
            mul_35,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_6_modules_mlp_modules_fc3_parameters_weight_,
            None,
        )
        mul_35 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_6_modules_mlp_modules_fc3_parameters_weight_ = (None)
        x_55 = silu_6 * linear_33
        silu_6 = linear_33 = None
        x_56 = torch._C._nn.linear(
            x_55,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_55 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_57 = x_54 + x_56
        x_54 = x_56 = None
        float_16 = x_57.float()
        pow_16 = float_16.pow(2)
        mean_15 = pow_16.mean(-1, keepdim=True)
        pow_16 = None
        add_30 = mean_15 + 1e-05
        mean_15 = None
        rsqrt_15 = torch.rsqrt(add_30)
        add_30 = None
        mul_37 = float_16 * rsqrt_15
        float_16 = rsqrt_15 = None
        output_15 = mul_37.type_as(x_57)
        mul_37 = None
        mul_38 = (
            output_15
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_7_modules_norm_1_parameters_weight_
        )
        output_15 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_7_modules_norm_1_parameters_weight_ = (None)
        linear_35 = torch._C._nn.linear(
            mul_38,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_,
            None,
        )
        mul_38 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_ = (None)
        reshape_14 = linear_35.reshape(1, 256, 3, 8, 128)
        linear_35 = None
        qkv_7 = reshape_14.permute(2, 0, 3, 1, 4)
        reshape_14 = None
        unbind_7 = qkv_7.unbind(0)
        qkv_7 = None
        q_7 = unbind_7[0]
        k_7 = unbind_7[1]
        v_7 = unbind_7[2]
        unbind_7 = None
        x_58 = torch._C._nn.scaled_dot_product_attention(q_7, k_7, v_7, is_causal=False)
        q_7 = k_7 = v_7 = None
        transpose_8 = x_58.transpose(1, 2)
        x_58 = None
        contiguous_7 = transpose_8.contiguous()
        transpose_8 = None
        x_59 = contiguous_7.reshape(1, 256, 1024)
        contiguous_7 = None
        x_60 = torch._C._nn.linear(
            x_59,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_,
            None,
        )
        x_59 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_ = (None)
        x_61 = torch.nn.functional.dropout(x_60, 0.0, False, False)
        x_60 = None
        x_62 = x_57 + x_61
        x_57 = x_61 = None
        float_17 = x_62.float()
        pow_17 = float_17.pow(2)
        mean_16 = pow_17.mean(-1, keepdim=True)
        pow_17 = None
        add_32 = mean_16 + 1e-05
        mean_16 = None
        rsqrt_16 = torch.rsqrt(add_32)
        add_32 = None
        mul_39 = float_17 * rsqrt_16
        float_17 = rsqrt_16 = None
        output_16 = mul_39.type_as(x_62)
        mul_39 = None
        mul_40 = (
            output_16
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_7_modules_norm_2_parameters_weight_
        )
        output_16 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_7_modules_norm_2_parameters_weight_ = (None)
        linear_37 = torch._C._nn.linear(
            mul_40,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = (
            None
        )
        silu_7 = torch.nn.functional.silu(linear_37)
        linear_37 = None
        linear_38 = torch._C._nn.linear(
            mul_40,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_7_modules_mlp_modules_fc3_parameters_weight_,
            None,
        )
        mul_40 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_7_modules_mlp_modules_fc3_parameters_weight_ = (None)
        x_63 = silu_7 * linear_38
        silu_7 = linear_38 = None
        x_64 = torch._C._nn.linear(
            x_63,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_63 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_65 = x_62 + x_64
        x_62 = x_64 = None
        float_18 = x_65.float()
        pow_18 = float_18.pow(2)
        mean_17 = pow_18.mean(-1, keepdim=True)
        pow_18 = None
        add_34 = mean_17 + 1e-05
        mean_17 = None
        rsqrt_17 = torch.rsqrt(add_34)
        add_34 = None
        mul_42 = float_18 * rsqrt_17
        float_18 = rsqrt_17 = None
        output_17 = mul_42.type_as(x_65)
        mul_42 = None
        mul_43 = (
            output_17
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_8_modules_norm_1_parameters_weight_
        )
        output_17 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_8_modules_norm_1_parameters_weight_ = (None)
        linear_40 = torch._C._nn.linear(
            mul_43,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_,
            None,
        )
        mul_43 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_ = (None)
        reshape_16 = linear_40.reshape(1, 256, 3, 8, 128)
        linear_40 = None
        qkv_8 = reshape_16.permute(2, 0, 3, 1, 4)
        reshape_16 = None
        unbind_8 = qkv_8.unbind(0)
        qkv_8 = None
        q_8 = unbind_8[0]
        k_8 = unbind_8[1]
        v_8 = unbind_8[2]
        unbind_8 = None
        x_66 = torch._C._nn.scaled_dot_product_attention(q_8, k_8, v_8, is_causal=False)
        q_8 = k_8 = v_8 = None
        transpose_9 = x_66.transpose(1, 2)
        x_66 = None
        contiguous_8 = transpose_9.contiguous()
        transpose_9 = None
        x_67 = contiguous_8.reshape(1, 256, 1024)
        contiguous_8 = None
        x_68 = torch._C._nn.linear(
            x_67,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_,
            None,
        )
        x_67 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_ = (None)
        x_69 = torch.nn.functional.dropout(x_68, 0.0, False, False)
        x_68 = None
        x_70 = x_65 + x_69
        x_65 = x_69 = None
        float_19 = x_70.float()
        pow_19 = float_19.pow(2)
        mean_18 = pow_19.mean(-1, keepdim=True)
        pow_19 = None
        add_36 = mean_18 + 1e-05
        mean_18 = None
        rsqrt_18 = torch.rsqrt(add_36)
        add_36 = None
        mul_44 = float_19 * rsqrt_18
        float_19 = rsqrt_18 = None
        output_18 = mul_44.type_as(x_70)
        mul_44 = None
        mul_45 = (
            output_18
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_8_modules_norm_2_parameters_weight_
        )
        output_18 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_8_modules_norm_2_parameters_weight_ = (None)
        linear_42 = torch._C._nn.linear(
            mul_45,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_ = (
            None
        )
        silu_8 = torch.nn.functional.silu(linear_42)
        linear_42 = None
        linear_43 = torch._C._nn.linear(
            mul_45,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_8_modules_mlp_modules_fc3_parameters_weight_,
            None,
        )
        mul_45 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_8_modules_mlp_modules_fc3_parameters_weight_ = (None)
        x_71 = silu_8 * linear_43
        silu_8 = linear_43 = None
        x_72 = torch._C._nn.linear(
            x_71,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_71 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_73 = x_70 + x_72
        x_70 = x_72 = None
        float_20 = x_73.float()
        pow_20 = float_20.pow(2)
        mean_19 = pow_20.mean(-1, keepdim=True)
        pow_20 = None
        add_38 = mean_19 + 1e-05
        mean_19 = None
        rsqrt_19 = torch.rsqrt(add_38)
        add_38 = None
        mul_47 = float_20 * rsqrt_19
        float_20 = rsqrt_19 = None
        output_19 = mul_47.type_as(x_73)
        mul_47 = None
        mul_48 = (
            output_19
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_9_modules_norm_1_parameters_weight_
        )
        output_19 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_9_modules_norm_1_parameters_weight_ = (None)
        linear_45 = torch._C._nn.linear(
            mul_48,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_,
            None,
        )
        mul_48 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_ = (None)
        reshape_18 = linear_45.reshape(1, 256, 3, 8, 128)
        linear_45 = None
        qkv_9 = reshape_18.permute(2, 0, 3, 1, 4)
        reshape_18 = None
        unbind_9 = qkv_9.unbind(0)
        qkv_9 = None
        q_9 = unbind_9[0]
        k_9 = unbind_9[1]
        v_9 = unbind_9[2]
        unbind_9 = None
        x_74 = torch._C._nn.scaled_dot_product_attention(q_9, k_9, v_9, is_causal=False)
        q_9 = k_9 = v_9 = None
        transpose_10 = x_74.transpose(1, 2)
        x_74 = None
        contiguous_9 = transpose_10.contiguous()
        transpose_10 = None
        x_75 = contiguous_9.reshape(1, 256, 1024)
        contiguous_9 = None
        x_76 = torch._C._nn.linear(
            x_75,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_,
            None,
        )
        x_75 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_ = (None)
        x_77 = torch.nn.functional.dropout(x_76, 0.0, False, False)
        x_76 = None
        x_78 = x_73 + x_77
        x_73 = x_77 = None
        float_21 = x_78.float()
        pow_21 = float_21.pow(2)
        mean_20 = pow_21.mean(-1, keepdim=True)
        pow_21 = None
        add_40 = mean_20 + 1e-05
        mean_20 = None
        rsqrt_20 = torch.rsqrt(add_40)
        add_40 = None
        mul_49 = float_21 * rsqrt_20
        float_21 = rsqrt_20 = None
        output_20 = mul_49.type_as(x_78)
        mul_49 = None
        mul_50 = (
            output_20
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_9_modules_norm_2_parameters_weight_
        )
        output_20 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_9_modules_norm_2_parameters_weight_ = (None)
        linear_47 = torch._C._nn.linear(
            mul_50,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_ = (
            None
        )
        silu_9 = torch.nn.functional.silu(linear_47)
        linear_47 = None
        linear_48 = torch._C._nn.linear(
            mul_50,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_9_modules_mlp_modules_fc3_parameters_weight_,
            None,
        )
        mul_50 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_9_modules_mlp_modules_fc3_parameters_weight_ = (None)
        x_79 = silu_9 * linear_48
        silu_9 = linear_48 = None
        x_80 = torch._C._nn.linear(
            x_79,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_79 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_81 = x_78 + x_80
        x_78 = x_80 = None
        float_22 = x_81.float()
        pow_22 = float_22.pow(2)
        mean_21 = pow_22.mean(-1, keepdim=True)
        pow_22 = None
        add_42 = mean_21 + 1e-05
        mean_21 = None
        rsqrt_21 = torch.rsqrt(add_42)
        add_42 = None
        mul_52 = float_22 * rsqrt_21
        float_22 = rsqrt_21 = None
        output_21 = mul_52.type_as(x_81)
        mul_52 = None
        mul_53 = (
            output_21
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_10_modules_norm_1_parameters_weight_
        )
        output_21 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_10_modules_norm_1_parameters_weight_ = (None)
        linear_50 = torch._C._nn.linear(
            mul_53,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_,
            None,
        )
        mul_53 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_ = (None)
        reshape_20 = linear_50.reshape(1, 256, 3, 8, 128)
        linear_50 = None
        qkv_10 = reshape_20.permute(2, 0, 3, 1, 4)
        reshape_20 = None
        unbind_10 = qkv_10.unbind(0)
        qkv_10 = None
        q_10 = unbind_10[0]
        k_10 = unbind_10[1]
        v_10 = unbind_10[2]
        unbind_10 = None
        x_82 = torch._C._nn.scaled_dot_product_attention(
            q_10, k_10, v_10, is_causal=False
        )
        q_10 = k_10 = v_10 = None
        transpose_11 = x_82.transpose(1, 2)
        x_82 = None
        contiguous_10 = transpose_11.contiguous()
        transpose_11 = None
        x_83 = contiguous_10.reshape(1, 256, 1024)
        contiguous_10 = None
        x_84 = torch._C._nn.linear(
            x_83,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_,
            None,
        )
        x_83 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_ = (None)
        x_85 = torch.nn.functional.dropout(x_84, 0.0, False, False)
        x_84 = None
        x_86 = x_81 + x_85
        x_81 = x_85 = None
        float_23 = x_86.float()
        pow_23 = float_23.pow(2)
        mean_22 = pow_23.mean(-1, keepdim=True)
        pow_23 = None
        add_44 = mean_22 + 1e-05
        mean_22 = None
        rsqrt_22 = torch.rsqrt(add_44)
        add_44 = None
        mul_54 = float_23 * rsqrt_22
        float_23 = rsqrt_22 = None
        output_22 = mul_54.type_as(x_86)
        mul_54 = None
        mul_55 = (
            output_22
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_10_modules_norm_2_parameters_weight_
        )
        output_22 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_10_modules_norm_2_parameters_weight_ = (None)
        linear_52 = torch._C._nn.linear(
            mul_55,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_ = (
            None
        )
        silu_10 = torch.nn.functional.silu(linear_52)
        linear_52 = None
        linear_53 = torch._C._nn.linear(
            mul_55,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_10_modules_mlp_modules_fc3_parameters_weight_,
            None,
        )
        mul_55 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_10_modules_mlp_modules_fc3_parameters_weight_ = (None)
        x_87 = silu_10 * linear_53
        silu_10 = linear_53 = None
        x_88 = torch._C._nn.linear(
            x_87,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_87 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_89 = x_86 + x_88
        x_86 = x_88 = None
        float_24 = x_89.float()
        pow_24 = float_24.pow(2)
        mean_23 = pow_24.mean(-1, keepdim=True)
        pow_24 = None
        add_46 = mean_23 + 1e-05
        mean_23 = None
        rsqrt_23 = torch.rsqrt(add_46)
        add_46 = None
        mul_57 = float_24 * rsqrt_23
        float_24 = rsqrt_23 = None
        output_23 = mul_57.type_as(x_89)
        mul_57 = None
        mul_58 = (
            output_23
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_11_modules_norm_1_parameters_weight_
        )
        output_23 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_11_modules_norm_1_parameters_weight_ = (None)
        linear_55 = torch._C._nn.linear(
            mul_58,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_,
            None,
        )
        mul_58 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_ = (None)
        reshape_22 = linear_55.reshape(1, 256, 3, 8, 128)
        linear_55 = None
        qkv_11 = reshape_22.permute(2, 0, 3, 1, 4)
        reshape_22 = None
        unbind_11 = qkv_11.unbind(0)
        qkv_11 = None
        q_11 = unbind_11[0]
        k_11 = unbind_11[1]
        v_11 = unbind_11[2]
        unbind_11 = None
        x_90 = torch._C._nn.scaled_dot_product_attention(
            q_11, k_11, v_11, is_causal=False
        )
        q_11 = k_11 = v_11 = None
        transpose_12 = x_90.transpose(1, 2)
        x_90 = None
        contiguous_11 = transpose_12.contiguous()
        transpose_12 = None
        x_91 = contiguous_11.reshape(1, 256, 1024)
        contiguous_11 = None
        x_92 = torch._C._nn.linear(
            x_91,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_,
            None,
        )
        x_91 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_ = (None)
        x_93 = torch.nn.functional.dropout(x_92, 0.0, False, False)
        x_92 = None
        x_94 = x_89 + x_93
        x_89 = x_93 = None
        float_25 = x_94.float()
        pow_25 = float_25.pow(2)
        mean_24 = pow_25.mean(-1, keepdim=True)
        pow_25 = None
        add_48 = mean_24 + 1e-05
        mean_24 = None
        rsqrt_24 = torch.rsqrt(add_48)
        add_48 = None
        mul_59 = float_25 * rsqrt_24
        float_25 = rsqrt_24 = None
        output_24 = mul_59.type_as(x_94)
        mul_59 = None
        mul_60 = (
            output_24
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_11_modules_norm_2_parameters_weight_
        )
        output_24 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_11_modules_norm_2_parameters_weight_ = (None)
        linear_57 = torch._C._nn.linear(
            mul_60,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_ = (
            None
        )
        silu_11 = torch.nn.functional.silu(linear_57)
        linear_57 = None
        linear_58 = torch._C._nn.linear(
            mul_60,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_11_modules_mlp_modules_fc3_parameters_weight_,
            None,
        )
        mul_60 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_11_modules_mlp_modules_fc3_parameters_weight_ = (None)
        x_95 = silu_11 * linear_58
        silu_11 = linear_58 = None
        x_96 = torch._C._nn.linear(
            x_95,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_95 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_97 = x_94 + x_96
        x_94 = x_96 = None
        float_26 = x_97.float()
        pow_26 = float_26.pow(2)
        mean_25 = pow_26.mean(-1, keepdim=True)
        pow_26 = None
        add_50 = mean_25 + 1e-05
        mean_25 = None
        rsqrt_25 = torch.rsqrt(add_50)
        add_50 = None
        mul_62 = float_26 * rsqrt_25
        float_26 = rsqrt_25 = None
        output_25 = mul_62.type_as(x_97)
        mul_62 = None
        mul_63 = (
            output_25
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_12_modules_norm_1_parameters_weight_
        )
        output_25 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_12_modules_norm_1_parameters_weight_ = (None)
        linear_60 = torch._C._nn.linear(
            mul_63,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_,
            None,
        )
        mul_63 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_ = (None)
        reshape_24 = linear_60.reshape(1, 256, 3, 8, 128)
        linear_60 = None
        qkv_12 = reshape_24.permute(2, 0, 3, 1, 4)
        reshape_24 = None
        unbind_12 = qkv_12.unbind(0)
        qkv_12 = None
        q_12 = unbind_12[0]
        k_12 = unbind_12[1]
        v_12 = unbind_12[2]
        unbind_12 = None
        x_98 = torch._C._nn.scaled_dot_product_attention(
            q_12, k_12, v_12, is_causal=False
        )
        q_12 = k_12 = v_12 = None
        transpose_13 = x_98.transpose(1, 2)
        x_98 = None
        contiguous_12 = transpose_13.contiguous()
        transpose_13 = None
        x_99 = contiguous_12.reshape(1, 256, 1024)
        contiguous_12 = None
        x_100 = torch._C._nn.linear(
            x_99,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_,
            None,
        )
        x_99 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_ = (None)
        x_101 = torch.nn.functional.dropout(x_100, 0.0, False, False)
        x_100 = None
        x_102 = x_97 + x_101
        x_97 = x_101 = None
        float_27 = x_102.float()
        pow_27 = float_27.pow(2)
        mean_26 = pow_27.mean(-1, keepdim=True)
        pow_27 = None
        add_52 = mean_26 + 1e-05
        mean_26 = None
        rsqrt_26 = torch.rsqrt(add_52)
        add_52 = None
        mul_64 = float_27 * rsqrt_26
        float_27 = rsqrt_26 = None
        output_26 = mul_64.type_as(x_102)
        mul_64 = None
        mul_65 = (
            output_26
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_12_modules_norm_2_parameters_weight_
        )
        output_26 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_12_modules_norm_2_parameters_weight_ = (None)
        linear_62 = torch._C._nn.linear(
            mul_65,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_ = (
            None
        )
        silu_12 = torch.nn.functional.silu(linear_62)
        linear_62 = None
        linear_63 = torch._C._nn.linear(
            mul_65,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_12_modules_mlp_modules_fc3_parameters_weight_,
            None,
        )
        mul_65 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_12_modules_mlp_modules_fc3_parameters_weight_ = (None)
        x_103 = silu_12 * linear_63
        silu_12 = linear_63 = None
        x_104 = torch._C._nn.linear(
            x_103,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_103 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_105 = x_102 + x_104
        x_102 = x_104 = None
        float_28 = x_105.float()
        pow_28 = float_28.pow(2)
        mean_27 = pow_28.mean(-1, keepdim=True)
        pow_28 = None
        add_54 = mean_27 + 1e-05
        mean_27 = None
        rsqrt_27 = torch.rsqrt(add_54)
        add_54 = None
        mul_67 = float_28 * rsqrt_27
        float_28 = rsqrt_27 = None
        output_27 = mul_67.type_as(x_105)
        mul_67 = None
        mul_68 = (
            output_27
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_13_modules_norm_1_parameters_weight_
        )
        output_27 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_13_modules_norm_1_parameters_weight_ = (None)
        linear_65 = torch._C._nn.linear(
            mul_68,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_,
            None,
        )
        mul_68 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_ = (None)
        reshape_26 = linear_65.reshape(1, 256, 3, 8, 128)
        linear_65 = None
        qkv_13 = reshape_26.permute(2, 0, 3, 1, 4)
        reshape_26 = None
        unbind_13 = qkv_13.unbind(0)
        qkv_13 = None
        q_13 = unbind_13[0]
        k_13 = unbind_13[1]
        v_13 = unbind_13[2]
        unbind_13 = None
        x_106 = torch._C._nn.scaled_dot_product_attention(
            q_13, k_13, v_13, is_causal=False
        )
        q_13 = k_13 = v_13 = None
        transpose_14 = x_106.transpose(1, 2)
        x_106 = None
        contiguous_13 = transpose_14.contiguous()
        transpose_14 = None
        x_107 = contiguous_13.reshape(1, 256, 1024)
        contiguous_13 = None
        x_108 = torch._C._nn.linear(
            x_107,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_,
            None,
        )
        x_107 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_ = (None)
        x_109 = torch.nn.functional.dropout(x_108, 0.0, False, False)
        x_108 = None
        x_110 = x_105 + x_109
        x_105 = x_109 = None
        float_29 = x_110.float()
        pow_29 = float_29.pow(2)
        mean_28 = pow_29.mean(-1, keepdim=True)
        pow_29 = None
        add_56 = mean_28 + 1e-05
        mean_28 = None
        rsqrt_28 = torch.rsqrt(add_56)
        add_56 = None
        mul_69 = float_29 * rsqrt_28
        float_29 = rsqrt_28 = None
        output_28 = mul_69.type_as(x_110)
        mul_69 = None
        mul_70 = (
            output_28
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_13_modules_norm_2_parameters_weight_
        )
        output_28 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_13_modules_norm_2_parameters_weight_ = (None)
        linear_67 = torch._C._nn.linear(
            mul_70,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_ = (
            None
        )
        silu_13 = torch.nn.functional.silu(linear_67)
        linear_67 = None
        linear_68 = torch._C._nn.linear(
            mul_70,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_13_modules_mlp_modules_fc3_parameters_weight_,
            None,
        )
        mul_70 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_13_modules_mlp_modules_fc3_parameters_weight_ = (None)
        x_111 = silu_13 * linear_68
        silu_13 = linear_68 = None
        x_112 = torch._C._nn.linear(
            x_111,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_111 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_113 = x_110 + x_112
        x_110 = x_112 = None
        float_30 = x_113.float()
        pow_30 = float_30.pow(2)
        mean_29 = pow_30.mean(-1, keepdim=True)
        pow_30 = None
        add_58 = mean_29 + 1e-05
        mean_29 = None
        rsqrt_29 = torch.rsqrt(add_58)
        add_58 = None
        mul_72 = float_30 * rsqrt_29
        float_30 = rsqrt_29 = None
        output_29 = mul_72.type_as(x_113)
        mul_72 = None
        mul_73 = (
            output_29
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_14_modules_norm_1_parameters_weight_
        )
        output_29 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_14_modules_norm_1_parameters_weight_ = (None)
        linear_70 = torch._C._nn.linear(
            mul_73,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_,
            None,
        )
        mul_73 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_ = (None)
        reshape_28 = linear_70.reshape(1, 256, 3, 8, 128)
        linear_70 = None
        qkv_14 = reshape_28.permute(2, 0, 3, 1, 4)
        reshape_28 = None
        unbind_14 = qkv_14.unbind(0)
        qkv_14 = None
        q_14 = unbind_14[0]
        k_14 = unbind_14[1]
        v_14 = unbind_14[2]
        unbind_14 = None
        x_114 = torch._C._nn.scaled_dot_product_attention(
            q_14, k_14, v_14, is_causal=False
        )
        q_14 = k_14 = v_14 = None
        transpose_15 = x_114.transpose(1, 2)
        x_114 = None
        contiguous_14 = transpose_15.contiguous()
        transpose_15 = None
        x_115 = contiguous_14.reshape(1, 256, 1024)
        contiguous_14 = None
        x_116 = torch._C._nn.linear(
            x_115,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_,
            None,
        )
        x_115 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_ = (None)
        x_117 = torch.nn.functional.dropout(x_116, 0.0, False, False)
        x_116 = None
        x_118 = x_113 + x_117
        x_113 = x_117 = None
        float_31 = x_118.float()
        pow_31 = float_31.pow(2)
        mean_30 = pow_31.mean(-1, keepdim=True)
        pow_31 = None
        add_60 = mean_30 + 1e-05
        mean_30 = None
        rsqrt_30 = torch.rsqrt(add_60)
        add_60 = None
        mul_74 = float_31 * rsqrt_30
        float_31 = rsqrt_30 = None
        output_30 = mul_74.type_as(x_118)
        mul_74 = None
        mul_75 = (
            output_30
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_14_modules_norm_2_parameters_weight_
        )
        output_30 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_14_modules_norm_2_parameters_weight_ = (None)
        linear_72 = torch._C._nn.linear(
            mul_75,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_ = (
            None
        )
        silu_14 = torch.nn.functional.silu(linear_72)
        linear_72 = None
        linear_73 = torch._C._nn.linear(
            mul_75,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_14_modules_mlp_modules_fc3_parameters_weight_,
            None,
        )
        mul_75 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_14_modules_mlp_modules_fc3_parameters_weight_ = (None)
        x_119 = silu_14 * linear_73
        silu_14 = linear_73 = None
        x_120 = torch._C._nn.linear(
            x_119,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_119 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_121 = x_118 + x_120
        x_118 = x_120 = None
        float_32 = x_121.float()
        pow_32 = float_32.pow(2)
        mean_31 = pow_32.mean(-1, keepdim=True)
        pow_32 = None
        add_62 = mean_31 + 1e-05
        mean_31 = None
        rsqrt_31 = torch.rsqrt(add_62)
        add_62 = None
        mul_77 = float_32 * rsqrt_31
        float_32 = rsqrt_31 = None
        output_31 = mul_77.type_as(x_121)
        mul_77 = None
        mul_78 = (
            output_31
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_15_modules_norm_1_parameters_weight_
        )
        output_31 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_15_modules_norm_1_parameters_weight_ = (None)
        linear_75 = torch._C._nn.linear(
            mul_78,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_,
            None,
        )
        mul_78 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_ = (None)
        reshape_30 = linear_75.reshape(1, 256, 3, 8, 128)
        linear_75 = None
        qkv_15 = reshape_30.permute(2, 0, 3, 1, 4)
        reshape_30 = None
        unbind_15 = qkv_15.unbind(0)
        qkv_15 = None
        q_15 = unbind_15[0]
        k_15 = unbind_15[1]
        v_15 = unbind_15[2]
        unbind_15 = None
        x_122 = torch._C._nn.scaled_dot_product_attention(
            q_15, k_15, v_15, is_causal=False
        )
        q_15 = k_15 = v_15 = None
        transpose_16 = x_122.transpose(1, 2)
        x_122 = None
        contiguous_15 = transpose_16.contiguous()
        transpose_16 = None
        x_123 = contiguous_15.reshape(1, 256, 1024)
        contiguous_15 = None
        x_124 = torch._C._nn.linear(
            x_123,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_,
            None,
        )
        x_123 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_ = (None)
        x_125 = torch.nn.functional.dropout(x_124, 0.0, False, False)
        x_124 = None
        x_126 = x_121 + x_125
        x_121 = x_125 = None
        float_33 = x_126.float()
        pow_33 = float_33.pow(2)
        mean_32 = pow_33.mean(-1, keepdim=True)
        pow_33 = None
        add_64 = mean_32 + 1e-05
        mean_32 = None
        rsqrt_32 = torch.rsqrt(add_64)
        add_64 = None
        mul_79 = float_33 * rsqrt_32
        float_33 = rsqrt_32 = None
        output_32 = mul_79.type_as(x_126)
        mul_79 = None
        mul_80 = (
            output_32
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_15_modules_norm_2_parameters_weight_
        )
        output_32 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_15_modules_norm_2_parameters_weight_ = (None)
        linear_77 = torch._C._nn.linear(
            mul_80,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_ = (
            None
        )
        silu_15 = torch.nn.functional.silu(linear_77)
        linear_77 = None
        linear_78 = torch._C._nn.linear(
            mul_80,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_15_modules_mlp_modules_fc3_parameters_weight_,
            None,
        )
        mul_80 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_15_modules_mlp_modules_fc3_parameters_weight_ = (None)
        x_127 = silu_15 * linear_78
        silu_15 = linear_78 = None
        x_128 = torch._C._nn.linear(
            x_127,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_127 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_129 = x_126 + x_128
        x_126 = x_128 = None
        float_34 = x_129.float()
        pow_34 = float_34.pow(2)
        mean_33 = pow_34.mean(-1, keepdim=True)
        pow_34 = None
        add_66 = mean_33 + 1e-05
        mean_33 = None
        rsqrt_33 = torch.rsqrt(add_66)
        add_66 = None
        mul_82 = float_34 * rsqrt_33
        float_34 = rsqrt_33 = None
        output_33 = mul_82.type_as(x_129)
        mul_82 = None
        mul_83 = (
            output_33
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_16_modules_norm_1_parameters_weight_
        )
        output_33 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_16_modules_norm_1_parameters_weight_ = (None)
        linear_80 = torch._C._nn.linear(
            mul_83,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_,
            None,
        )
        mul_83 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_ = (None)
        reshape_32 = linear_80.reshape(1, 256, 3, 8, 128)
        linear_80 = None
        qkv_16 = reshape_32.permute(2, 0, 3, 1, 4)
        reshape_32 = None
        unbind_16 = qkv_16.unbind(0)
        qkv_16 = None
        q_16 = unbind_16[0]
        k_16 = unbind_16[1]
        v_16 = unbind_16[2]
        unbind_16 = None
        x_130 = torch._C._nn.scaled_dot_product_attention(
            q_16, k_16, v_16, is_causal=False
        )
        q_16 = k_16 = v_16 = None
        transpose_17 = x_130.transpose(1, 2)
        x_130 = None
        contiguous_16 = transpose_17.contiguous()
        transpose_17 = None
        x_131 = contiguous_16.reshape(1, 256, 1024)
        contiguous_16 = None
        x_132 = torch._C._nn.linear(
            x_131,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_,
            None,
        )
        x_131 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_ = (None)
        x_133 = torch.nn.functional.dropout(x_132, 0.0, False, False)
        x_132 = None
        x_134 = x_129 + x_133
        x_129 = x_133 = None
        float_35 = x_134.float()
        pow_35 = float_35.pow(2)
        mean_34 = pow_35.mean(-1, keepdim=True)
        pow_35 = None
        add_68 = mean_34 + 1e-05
        mean_34 = None
        rsqrt_34 = torch.rsqrt(add_68)
        add_68 = None
        mul_84 = float_35 * rsqrt_34
        float_35 = rsqrt_34 = None
        output_34 = mul_84.type_as(x_134)
        mul_84 = None
        mul_85 = (
            output_34
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_16_modules_norm_2_parameters_weight_
        )
        output_34 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_16_modules_norm_2_parameters_weight_ = (None)
        linear_82 = torch._C._nn.linear(
            mul_85,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_ = (
            None
        )
        silu_16 = torch.nn.functional.silu(linear_82)
        linear_82 = None
        linear_83 = torch._C._nn.linear(
            mul_85,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_16_modules_mlp_modules_fc3_parameters_weight_,
            None,
        )
        mul_85 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_16_modules_mlp_modules_fc3_parameters_weight_ = (None)
        x_135 = silu_16 * linear_83
        silu_16 = linear_83 = None
        x_136 = torch._C._nn.linear(
            x_135,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_135 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_137 = x_134 + x_136
        x_134 = x_136 = None
        float_36 = x_137.float()
        pow_36 = float_36.pow(2)
        mean_35 = pow_36.mean(-1, keepdim=True)
        pow_36 = None
        add_70 = mean_35 + 1e-05
        mean_35 = None
        rsqrt_35 = torch.rsqrt(add_70)
        add_70 = None
        mul_87 = float_36 * rsqrt_35
        float_36 = rsqrt_35 = None
        output_35 = mul_87.type_as(x_137)
        mul_87 = None
        mul_88 = (
            output_35
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_17_modules_norm_1_parameters_weight_
        )
        output_35 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_17_modules_norm_1_parameters_weight_ = (None)
        linear_85 = torch._C._nn.linear(
            mul_88,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_,
            None,
        )
        mul_88 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_ = (None)
        reshape_34 = linear_85.reshape(1, 256, 3, 8, 128)
        linear_85 = None
        qkv_17 = reshape_34.permute(2, 0, 3, 1, 4)
        reshape_34 = None
        unbind_17 = qkv_17.unbind(0)
        qkv_17 = None
        q_17 = unbind_17[0]
        k_17 = unbind_17[1]
        v_17 = unbind_17[2]
        unbind_17 = None
        x_138 = torch._C._nn.scaled_dot_product_attention(
            q_17, k_17, v_17, is_causal=False
        )
        q_17 = k_17 = v_17 = None
        transpose_18 = x_138.transpose(1, 2)
        x_138 = None
        contiguous_17 = transpose_18.contiguous()
        transpose_18 = None
        x_139 = contiguous_17.reshape(1, 256, 1024)
        contiguous_17 = None
        x_140 = torch._C._nn.linear(
            x_139,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_,
            None,
        )
        x_139 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_ = (None)
        x_141 = torch.nn.functional.dropout(x_140, 0.0, False, False)
        x_140 = None
        x_142 = x_137 + x_141
        x_137 = x_141 = None
        float_37 = x_142.float()
        pow_37 = float_37.pow(2)
        mean_36 = pow_37.mean(-1, keepdim=True)
        pow_37 = None
        add_72 = mean_36 + 1e-05
        mean_36 = None
        rsqrt_36 = torch.rsqrt(add_72)
        add_72 = None
        mul_89 = float_37 * rsqrt_36
        float_37 = rsqrt_36 = None
        output_36 = mul_89.type_as(x_142)
        mul_89 = None
        mul_90 = (
            output_36
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_17_modules_norm_2_parameters_weight_
        )
        output_36 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_17_modules_norm_2_parameters_weight_ = (None)
        linear_87 = torch._C._nn.linear(
            mul_90,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_ = (
            None
        )
        silu_17 = torch.nn.functional.silu(linear_87)
        linear_87 = None
        linear_88 = torch._C._nn.linear(
            mul_90,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_17_modules_mlp_modules_fc3_parameters_weight_,
            None,
        )
        mul_90 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_17_modules_mlp_modules_fc3_parameters_weight_ = (None)
        x_143 = silu_17 * linear_88
        silu_17 = linear_88 = None
        x_144 = torch._C._nn.linear(
            x_143,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_143 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_145 = x_142 + x_144
        x_142 = x_144 = None
        float_38 = x_145.float()
        pow_38 = float_38.pow(2)
        mean_37 = pow_38.mean(-1, keepdim=True)
        pow_38 = None
        add_74 = mean_37 + 1e-05
        mean_37 = None
        rsqrt_37 = torch.rsqrt(add_74)
        add_74 = None
        mul_92 = float_38 * rsqrt_37
        float_38 = rsqrt_37 = None
        output_37 = mul_92.type_as(x_145)
        mul_92 = None
        mul_93 = (
            output_37
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_18_modules_norm_1_parameters_weight_
        )
        output_37 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_18_modules_norm_1_parameters_weight_ = (None)
        linear_90 = torch._C._nn.linear(
            mul_93,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_,
            None,
        )
        mul_93 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_ = (None)
        reshape_36 = linear_90.reshape(1, 256, 3, 8, 128)
        linear_90 = None
        qkv_18 = reshape_36.permute(2, 0, 3, 1, 4)
        reshape_36 = None
        unbind_18 = qkv_18.unbind(0)
        qkv_18 = None
        q_18 = unbind_18[0]
        k_18 = unbind_18[1]
        v_18 = unbind_18[2]
        unbind_18 = None
        x_146 = torch._C._nn.scaled_dot_product_attention(
            q_18, k_18, v_18, is_causal=False
        )
        q_18 = k_18 = v_18 = None
        transpose_19 = x_146.transpose(1, 2)
        x_146 = None
        contiguous_18 = transpose_19.contiguous()
        transpose_19 = None
        x_147 = contiguous_18.reshape(1, 256, 1024)
        contiguous_18 = None
        x_148 = torch._C._nn.linear(
            x_147,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_,
            None,
        )
        x_147 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_ = (None)
        x_149 = torch.nn.functional.dropout(x_148, 0.0, False, False)
        x_148 = None
        x_150 = x_145 + x_149
        x_145 = x_149 = None
        float_39 = x_150.float()
        pow_39 = float_39.pow(2)
        mean_38 = pow_39.mean(-1, keepdim=True)
        pow_39 = None
        add_76 = mean_38 + 1e-05
        mean_38 = None
        rsqrt_38 = torch.rsqrt(add_76)
        add_76 = None
        mul_94 = float_39 * rsqrt_38
        float_39 = rsqrt_38 = None
        output_38 = mul_94.type_as(x_150)
        mul_94 = None
        mul_95 = (
            output_38
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_18_modules_norm_2_parameters_weight_
        )
        output_38 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_18_modules_norm_2_parameters_weight_ = (None)
        linear_92 = torch._C._nn.linear(
            mul_95,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_ = (
            None
        )
        silu_18 = torch.nn.functional.silu(linear_92)
        linear_92 = None
        linear_93 = torch._C._nn.linear(
            mul_95,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_18_modules_mlp_modules_fc3_parameters_weight_,
            None,
        )
        mul_95 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_18_modules_mlp_modules_fc3_parameters_weight_ = (None)
        x_151 = silu_18 * linear_93
        silu_18 = linear_93 = None
        x_152 = torch._C._nn.linear(
            x_151,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_151 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_153 = x_150 + x_152
        x_150 = x_152 = None
        float_40 = x_153.float()
        pow_40 = float_40.pow(2)
        mean_39 = pow_40.mean(-1, keepdim=True)
        pow_40 = None
        add_78 = mean_39 + 1e-05
        mean_39 = None
        rsqrt_39 = torch.rsqrt(add_78)
        add_78 = None
        mul_97 = float_40 * rsqrt_39
        float_40 = rsqrt_39 = None
        output_39 = mul_97.type_as(x_153)
        mul_97 = None
        mul_98 = (
            output_39
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_19_modules_norm_1_parameters_weight_
        )
        output_39 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_19_modules_norm_1_parameters_weight_ = (None)
        linear_95 = torch._C._nn.linear(
            mul_98,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_,
            None,
        )
        mul_98 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_ = (None)
        reshape_38 = linear_95.reshape(1, 256, 3, 8, 128)
        linear_95 = None
        qkv_19 = reshape_38.permute(2, 0, 3, 1, 4)
        reshape_38 = None
        unbind_19 = qkv_19.unbind(0)
        qkv_19 = None
        q_19 = unbind_19[0]
        k_19 = unbind_19[1]
        v_19 = unbind_19[2]
        unbind_19 = None
        x_154 = torch._C._nn.scaled_dot_product_attention(
            q_19, k_19, v_19, is_causal=False
        )
        q_19 = k_19 = v_19 = None
        transpose_20 = x_154.transpose(1, 2)
        x_154 = None
        contiguous_19 = transpose_20.contiguous()
        transpose_20 = None
        x_155 = contiguous_19.reshape(1, 256, 1024)
        contiguous_19 = None
        x_156 = torch._C._nn.linear(
            x_155,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_,
            None,
        )
        x_155 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_ = (None)
        x_157 = torch.nn.functional.dropout(x_156, 0.0, False, False)
        x_156 = None
        x_158 = x_153 + x_157
        x_153 = x_157 = None
        float_41 = x_158.float()
        pow_41 = float_41.pow(2)
        mean_40 = pow_41.mean(-1, keepdim=True)
        pow_41 = None
        add_80 = mean_40 + 1e-05
        mean_40 = None
        rsqrt_40 = torch.rsqrt(add_80)
        add_80 = None
        mul_99 = float_41 * rsqrt_40
        float_41 = rsqrt_40 = None
        output_40 = mul_99.type_as(x_158)
        mul_99 = None
        mul_100 = (
            output_40
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_19_modules_norm_2_parameters_weight_
        )
        output_40 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_19_modules_norm_2_parameters_weight_ = (None)
        linear_97 = torch._C._nn.linear(
            mul_100,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_ = (
            None
        )
        silu_19 = torch.nn.functional.silu(linear_97)
        linear_97 = None
        linear_98 = torch._C._nn.linear(
            mul_100,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_19_modules_mlp_modules_fc3_parameters_weight_,
            None,
        )
        mul_100 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_19_modules_mlp_modules_fc3_parameters_weight_ = (None)
        x_159 = silu_19 * linear_98
        silu_19 = linear_98 = None
        x_160 = torch._C._nn.linear(
            x_159,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_159 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_161 = x_158 + x_160
        x_158 = x_160 = None
        float_42 = x_161.float()
        pow_42 = float_42.pow(2)
        mean_41 = pow_42.mean(-1, keepdim=True)
        pow_42 = None
        add_82 = mean_41 + 1e-05
        mean_41 = None
        rsqrt_41 = torch.rsqrt(add_82)
        add_82 = None
        mul_102 = float_42 * rsqrt_41
        float_42 = rsqrt_41 = None
        output_41 = mul_102.type_as(x_161)
        mul_102 = None
        mul_103 = (
            output_41
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_20_modules_norm_1_parameters_weight_
        )
        output_41 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_20_modules_norm_1_parameters_weight_ = (None)
        linear_100 = torch._C._nn.linear(
            mul_103,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_,
            None,
        )
        mul_103 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_ = (None)
        reshape_40 = linear_100.reshape(1, 256, 3, 8, 128)
        linear_100 = None
        qkv_20 = reshape_40.permute(2, 0, 3, 1, 4)
        reshape_40 = None
        unbind_20 = qkv_20.unbind(0)
        qkv_20 = None
        q_20 = unbind_20[0]
        k_20 = unbind_20[1]
        v_20 = unbind_20[2]
        unbind_20 = None
        x_162 = torch._C._nn.scaled_dot_product_attention(
            q_20, k_20, v_20, is_causal=False
        )
        q_20 = k_20 = v_20 = None
        transpose_21 = x_162.transpose(1, 2)
        x_162 = None
        contiguous_20 = transpose_21.contiguous()
        transpose_21 = None
        x_163 = contiguous_20.reshape(1, 256, 1024)
        contiguous_20 = None
        x_164 = torch._C._nn.linear(
            x_163,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_,
            None,
        )
        x_163 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_ = (None)
        x_165 = torch.nn.functional.dropout(x_164, 0.0, False, False)
        x_164 = None
        x_166 = x_161 + x_165
        x_161 = x_165 = None
        float_43 = x_166.float()
        pow_43 = float_43.pow(2)
        mean_42 = pow_43.mean(-1, keepdim=True)
        pow_43 = None
        add_84 = mean_42 + 1e-05
        mean_42 = None
        rsqrt_42 = torch.rsqrt(add_84)
        add_84 = None
        mul_104 = float_43 * rsqrt_42
        float_43 = rsqrt_42 = None
        output_42 = mul_104.type_as(x_166)
        mul_104 = None
        mul_105 = (
            output_42
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_20_modules_norm_2_parameters_weight_
        )
        output_42 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_20_modules_norm_2_parameters_weight_ = (None)
        linear_102 = torch._C._nn.linear(
            mul_105,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_ = (
            None
        )
        silu_20 = torch.nn.functional.silu(linear_102)
        linear_102 = None
        linear_103 = torch._C._nn.linear(
            mul_105,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_20_modules_mlp_modules_fc3_parameters_weight_,
            None,
        )
        mul_105 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_20_modules_mlp_modules_fc3_parameters_weight_ = (None)
        x_167 = silu_20 * linear_103
        silu_20 = linear_103 = None
        x_168 = torch._C._nn.linear(
            x_167,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_167 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_169 = x_166 + x_168
        x_166 = x_168 = None
        float_44 = x_169.float()
        pow_44 = float_44.pow(2)
        mean_43 = pow_44.mean(-1, keepdim=True)
        pow_44 = None
        add_86 = mean_43 + 1e-05
        mean_43 = None
        rsqrt_43 = torch.rsqrt(add_86)
        add_86 = None
        mul_107 = float_44 * rsqrt_43
        float_44 = rsqrt_43 = None
        output_43 = mul_107.type_as(x_169)
        mul_107 = None
        mul_108 = (
            output_43
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_21_modules_norm_1_parameters_weight_
        )
        output_43 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_21_modules_norm_1_parameters_weight_ = (None)
        linear_105 = torch._C._nn.linear(
            mul_108,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_,
            None,
        )
        mul_108 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_ = (None)
        reshape_42 = linear_105.reshape(1, 256, 3, 8, 128)
        linear_105 = None
        qkv_21 = reshape_42.permute(2, 0, 3, 1, 4)
        reshape_42 = None
        unbind_21 = qkv_21.unbind(0)
        qkv_21 = None
        q_21 = unbind_21[0]
        k_21 = unbind_21[1]
        v_21 = unbind_21[2]
        unbind_21 = None
        x_170 = torch._C._nn.scaled_dot_product_attention(
            q_21, k_21, v_21, is_causal=False
        )
        q_21 = k_21 = v_21 = None
        transpose_22 = x_170.transpose(1, 2)
        x_170 = None
        contiguous_21 = transpose_22.contiguous()
        transpose_22 = None
        x_171 = contiguous_21.reshape(1, 256, 1024)
        contiguous_21 = None
        x_172 = torch._C._nn.linear(
            x_171,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_,
            None,
        )
        x_171 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_ = (None)
        x_173 = torch.nn.functional.dropout(x_172, 0.0, False, False)
        x_172 = None
        x_174 = x_169 + x_173
        x_169 = x_173 = None
        float_45 = x_174.float()
        pow_45 = float_45.pow(2)
        mean_44 = pow_45.mean(-1, keepdim=True)
        pow_45 = None
        add_88 = mean_44 + 1e-05
        mean_44 = None
        rsqrt_44 = torch.rsqrt(add_88)
        add_88 = None
        mul_109 = float_45 * rsqrt_44
        float_45 = rsqrt_44 = None
        output_44 = mul_109.type_as(x_174)
        mul_109 = None
        mul_110 = (
            output_44
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_21_modules_norm_2_parameters_weight_
        )
        output_44 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_21_modules_norm_2_parameters_weight_ = (None)
        linear_107 = torch._C._nn.linear(
            mul_110,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_ = (
            None
        )
        silu_21 = torch.nn.functional.silu(linear_107)
        linear_107 = None
        linear_108 = torch._C._nn.linear(
            mul_110,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_21_modules_mlp_modules_fc3_parameters_weight_,
            None,
        )
        mul_110 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_21_modules_mlp_modules_fc3_parameters_weight_ = (None)
        x_175 = silu_21 * linear_108
        silu_21 = linear_108 = None
        x_176 = torch._C._nn.linear(
            x_175,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_175 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_177 = x_174 + x_176
        x_174 = x_176 = None
        float_46 = x_177.float()
        pow_46 = float_46.pow(2)
        mean_45 = pow_46.mean(-1, keepdim=True)
        pow_46 = None
        add_90 = mean_45 + 1e-05
        mean_45 = None
        rsqrt_45 = torch.rsqrt(add_90)
        add_90 = None
        mul_112 = float_46 * rsqrt_45
        float_46 = rsqrt_45 = None
        output_45 = mul_112.type_as(x_177)
        mul_112 = None
        mul_113 = (
            output_45
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_22_modules_norm_1_parameters_weight_
        )
        output_45 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_22_modules_norm_1_parameters_weight_ = (None)
        linear_110 = torch._C._nn.linear(
            mul_113,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_,
            None,
        )
        mul_113 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_ = (None)
        reshape_44 = linear_110.reshape(1, 256, 3, 8, 128)
        linear_110 = None
        qkv_22 = reshape_44.permute(2, 0, 3, 1, 4)
        reshape_44 = None
        unbind_22 = qkv_22.unbind(0)
        qkv_22 = None
        q_22 = unbind_22[0]
        k_22 = unbind_22[1]
        v_22 = unbind_22[2]
        unbind_22 = None
        x_178 = torch._C._nn.scaled_dot_product_attention(
            q_22, k_22, v_22, is_causal=False
        )
        q_22 = k_22 = v_22 = None
        transpose_23 = x_178.transpose(1, 2)
        x_178 = None
        contiguous_22 = transpose_23.contiguous()
        transpose_23 = None
        x_179 = contiguous_22.reshape(1, 256, 1024)
        contiguous_22 = None
        x_180 = torch._C._nn.linear(
            x_179,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_,
            None,
        )
        x_179 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_ = (None)
        x_181 = torch.nn.functional.dropout(x_180, 0.0, False, False)
        x_180 = None
        x_182 = x_177 + x_181
        x_177 = x_181 = None
        float_47 = x_182.float()
        pow_47 = float_47.pow(2)
        mean_46 = pow_47.mean(-1, keepdim=True)
        pow_47 = None
        add_92 = mean_46 + 1e-05
        mean_46 = None
        rsqrt_46 = torch.rsqrt(add_92)
        add_92 = None
        mul_114 = float_47 * rsqrt_46
        float_47 = rsqrt_46 = None
        output_46 = mul_114.type_as(x_182)
        mul_114 = None
        mul_115 = (
            output_46
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_22_modules_norm_2_parameters_weight_
        )
        output_46 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_22_modules_norm_2_parameters_weight_ = (None)
        linear_112 = torch._C._nn.linear(
            mul_115,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_ = (
            None
        )
        silu_22 = torch.nn.functional.silu(linear_112)
        linear_112 = None
        linear_113 = torch._C._nn.linear(
            mul_115,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_22_modules_mlp_modules_fc3_parameters_weight_,
            None,
        )
        mul_115 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_22_modules_mlp_modules_fc3_parameters_weight_ = (None)
        x_183 = silu_22 * linear_113
        silu_22 = linear_113 = None
        x_184 = torch._C._nn.linear(
            x_183,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_183 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_185 = x_182 + x_184
        x_182 = x_184 = None
        float_48 = x_185.float()
        pow_48 = float_48.pow(2)
        mean_47 = pow_48.mean(-1, keepdim=True)
        pow_48 = None
        add_94 = mean_47 + 1e-05
        mean_47 = None
        rsqrt_47 = torch.rsqrt(add_94)
        add_94 = None
        mul_117 = float_48 * rsqrt_47
        float_48 = rsqrt_47 = None
        output_47 = mul_117.type_as(x_185)
        mul_117 = None
        mul_118 = (
            output_47
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_23_modules_norm_1_parameters_weight_
        )
        output_47 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_23_modules_norm_1_parameters_weight_ = (None)
        linear_115 = torch._C._nn.linear(
            mul_118,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_,
            None,
        )
        mul_118 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_ = (None)
        reshape_46 = linear_115.reshape(1, 256, 3, 8, 128)
        linear_115 = None
        qkv_23 = reshape_46.permute(2, 0, 3, 1, 4)
        reshape_46 = None
        unbind_23 = qkv_23.unbind(0)
        qkv_23 = None
        q_23 = unbind_23[0]
        k_23 = unbind_23[1]
        v_23 = unbind_23[2]
        unbind_23 = None
        x_186 = torch._C._nn.scaled_dot_product_attention(
            q_23, k_23, v_23, is_causal=False
        )
        q_23 = k_23 = v_23 = None
        transpose_24 = x_186.transpose(1, 2)
        x_186 = None
        contiguous_23 = transpose_24.contiguous()
        transpose_24 = None
        x_187 = contiguous_23.reshape(1, 256, 1024)
        contiguous_23 = None
        x_188 = torch._C._nn.linear(
            x_187,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_,
            None,
        )
        x_187 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_ = (None)
        x_189 = torch.nn.functional.dropout(x_188, 0.0, False, False)
        x_188 = None
        x_190 = x_185 + x_189
        x_185 = x_189 = None
        float_49 = x_190.float()
        pow_49 = float_49.pow(2)
        mean_48 = pow_49.mean(-1, keepdim=True)
        pow_49 = None
        add_96 = mean_48 + 1e-05
        mean_48 = None
        rsqrt_48 = torch.rsqrt(add_96)
        add_96 = None
        mul_119 = float_49 * rsqrt_48
        float_49 = rsqrt_48 = None
        output_48 = mul_119.type_as(x_190)
        mul_119 = None
        mul_120 = (
            output_48
            * l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_23_modules_norm_2_parameters_weight_
        )
        output_48 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_23_modules_norm_2_parameters_weight_ = (None)
        linear_117 = torch._C._nn.linear(
            mul_120,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_ = (
            None
        )
        silu_23 = torch.nn.functional.silu(linear_117)
        linear_117 = None
        linear_118 = torch._C._nn.linear(
            mul_120,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_23_modules_mlp_modules_fc3_parameters_weight_,
            None,
        )
        mul_120 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_23_modules_mlp_modules_fc3_parameters_weight_ = (None)
        x_191 = silu_23 * linear_118
        silu_23 = linear_118 = None
        x_192 = torch._C._nn.linear(
            x_191,
            l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_191 = l_self_modules_image_encoder_modules_trunk_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_193 = x_190 + x_192
        x_190 = x_192 = None
        float_50 = x_193.float()
        pow_50 = float_50.pow(2)
        mean_49 = pow_50.mean(-1, keepdim=True)
        pow_50 = None
        add_98 = mean_49 + 1e-05
        mean_49 = None
        rsqrt_49 = torch.rsqrt(add_98)
        add_98 = None
        mul_122 = float_50 * rsqrt_49
        float_50 = rsqrt_49 = None
        output_49 = mul_122.type_as(x_193)
        mul_122 = x_193 = None
        tokens_1 = (
            output_49
            * l_self_modules_image_encoder_modules_trunk_modules_post_trunk_norm_parameters_weight_
        )
        output_49 = l_self_modules_image_encoder_modules_trunk_modules_post_trunk_norm_parameters_weight_ = (None)
        cls_token = (
            l_self_modules_image_encoder_modules_head_parameters_cls_token_.expand(
                1, -1, -1
            )
        )
        l_self_modules_image_encoder_modules_head_parameters_cls_token_ = None
        reshape_48 = cls_token.reshape(1, 1, 8, 128)
        cls_token = None
        q_24 = reshape_48.permute(0, 2, 1, 3)
        reshape_48 = None
        linear_120 = torch._C._nn.linear(
            tokens_1,
            l_self_modules_image_encoder_modules_head_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_image_encoder_modules_head_modules_k_parameters_weight_ = None
        reshape_49 = linear_120.reshape(1, 256, 8, 128)
        linear_120 = None
        k_24 = reshape_49.permute(0, 2, 1, 3)
        reshape_49 = None
        linear_121 = torch._C._nn.linear(
            tokens_1,
            l_self_modules_image_encoder_modules_head_modules_v_parameters_weight_,
            None,
        )
        tokens_1 = (
            l_self_modules_image_encoder_modules_head_modules_v_parameters_weight_
        ) = None
        reshape_50 = linear_121.reshape(1, 256, 8, 128)
        linear_121 = None
        v_24 = reshape_50.permute(0, 2, 1, 3)
        reshape_50 = None
        x_cls = torch._C._nn.scaled_dot_product_attention(q_24, k_24, v_24)
        q_24 = k_24 = v_24 = None
        transpose_25 = x_cls.transpose(1, 2)
        x_cls = None
        x_cls_1 = transpose_25.reshape(1, 1, 1024)
        transpose_25 = None
        x_cls_2 = x_cls_1.mean(dim=1)
        x_cls_1 = None
        out = torch._C._nn.linear(
            x_cls_2,
            l_self_modules_image_encoder_modules_head_modules_linear_parameters_weight_,
            l_self_modules_image_encoder_modules_head_modules_linear_parameters_bias_,
        )
        x_cls_2 = (
            l_self_modules_image_encoder_modules_head_modules_linear_parameters_weight_
        ) = (
            l_self_modules_image_encoder_modules_head_modules_linear_parameters_bias_
        ) = None
        image_features = torch._C._nn.linear(
            out, l_self_modules_image_projector_parameters_weight_, None
        )
        l_self_modules_image_projector_parameters_weight_ = None
        image_features_1 = torch.nn.functional.normalize(image_features, p=2, dim=-1)
        image_features = None
        eos_token_mask = l_input_ids_.__eq__(49407)
        tokens_2 = torch.nn.functional.embedding(
            l_input_ids_,
            l_self_modules_text_encoder_modules_preprocessor_modules_text_embedding_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        l_input_ids_ = l_self_modules_text_encoder_modules_preprocessor_modules_text_embedding_parameters_weight_ = (None)
        getitem_73 = tokens_2[(slice(None, None, None), slice(None, 7, None))]
        tokens_2 = None
        getitem_74 = l_self_modules_text_encoder_modules_preprocessor_parameters_positional_embedding_[
            slice(None, 7, None)
        ]
        l_self_modules_text_encoder_modules_preprocessor_parameters_positional_embedding_ = (
            None
        )
        unsqueeze = getitem_74.unsqueeze(0)
        getitem_74 = None
        tokens_3 = getitem_73 + unsqueeze
        getitem_73 = unsqueeze = None
        float_51 = tokens_3.float()
        pow_51 = float_51.pow(2)
        mean_51 = pow_51.mean(-1, keepdim=True)
        pow_51 = None
        add_100 = mean_51 + 1e-05
        mean_51 = None
        rsqrt_50 = torch.rsqrt(add_100)
        add_100 = None
        mul_124 = float_51 * rsqrt_50
        float_51 = rsqrt_50 = None
        output_50 = mul_124.type_as(tokens_3)
        mul_124 = None
        mul_125 = (
            output_50
            * l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_0_modules_norm_1_parameters_weight_
        )
        output_50 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_0_modules_norm_1_parameters_weight_ = (None)
        linear_124 = torch._C._nn.linear(
            mul_125,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_,
            None,
        )
        mul_125 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_ = (None)
        reshape_52 = linear_124.reshape(2, 7, 3, 6, 128)
        linear_124 = None
        qkv_24 = reshape_52.permute(2, 0, 3, 1, 4)
        reshape_52 = None
        unbind_24 = qkv_24.unbind(0)
        qkv_24 = None
        q_25 = unbind_24[0]
        k_25 = unbind_24[1]
        v_25 = unbind_24[2]
        unbind_24 = None
        mask = torch.full(
            (7, 7), -3.4028234663852886e38, device=device(type="cuda", index=0)
        )
        mask_cond = torch.arange(7, device=device(type="cuda", index=0))
        add_101 = mask_cond + 1
        view = add_101.view(7, 1)
        add_101 = None
        lt = mask_cond < view
        mask_cond = view = None
        masked_fill_ = mask.masked_fill_(lt, 0)
        lt = masked_fill_ = None
        mask_1 = mask.to(torch.float32)
        mask = None
        getitem_78 = mask_1[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        mask_1 = None
        causal_4d_mask = getitem_78.expand(2, 1, 7, 7)
        getitem_78 = None
        getitem_79 = l_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        expand_2 = getitem_79.expand(2, 1, 7, 7)
        getitem_79 = None
        expanded_mask = expand_2.to(torch.float32)
        expand_2 = None
        tensor = torch.tensor(1.0, dtype=torch.float32)
        inverted_mask = tensor - expanded_mask
        tensor = expanded_mask = None
        to_3 = inverted_mask.to(torch.bool)
        masked_fill = inverted_mask.masked_fill(to_3, -3.4028234663852886e38)
        inverted_mask = to_3 = None
        expanded_attn_mask = masked_fill.to(device(type="cuda", index=0))
        masked_fill = None
        bool_1 = expanded_attn_mask.bool()
        expanded_attn_mask = None
        expanded_attn_mask_1 = causal_4d_mask.masked_fill(
            bool_1, -3.4028234663852886e38
        )
        causal_4d_mask = bool_1 = None
        x_194 = torch._C._nn.scaled_dot_product_attention(
            q_25, k_25, v_25, attn_mask=expanded_attn_mask_1
        )
        q_25 = k_25 = v_25 = expanded_attn_mask_1 = None
        transpose_26 = x_194.transpose(1, 2)
        x_194 = None
        contiguous_24 = transpose_26.contiguous()
        transpose_26 = None
        x_195 = contiguous_24.reshape(2, 7, 768)
        contiguous_24 = None
        x_196 = torch._C._nn.linear(
            x_195,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_,
            None,
        )
        x_195 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_ = (None)
        x_197 = torch.nn.functional.dropout(x_196, 0.0, False, False)
        x_196 = None
        x_198 = tokens_3 + x_197
        tokens_3 = x_197 = None
        float_52 = x_198.float()
        pow_52 = float_52.pow(2)
        mean_52 = pow_52.mean(-1, keepdim=True)
        pow_52 = None
        add_103 = mean_52 + 1e-05
        mean_52 = None
        rsqrt_51 = torch.rsqrt(add_103)
        add_103 = None
        mul_126 = float_52 * rsqrt_51
        float_52 = rsqrt_51 = None
        output_51 = mul_126.type_as(x_198)
        mul_126 = None
        mul_127 = (
            output_51
            * l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_0_modules_norm_2_parameters_weight_
        )
        output_51 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_0_modules_norm_2_parameters_weight_ = (None)
        linear_126 = torch._C._nn.linear(
            mul_127,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = (
            None
        )
        silu_24 = torch.nn.functional.silu(linear_126)
        linear_126 = None
        linear_127 = torch._C._nn.linear(
            mul_127,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_0_modules_mlp_modules_fc3_parameters_weight_,
            None,
        )
        mul_127 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_0_modules_mlp_modules_fc3_parameters_weight_ = (None)
        x_199 = silu_24 * linear_127
        silu_24 = linear_127 = None
        x_200 = torch._C._nn.linear(
            x_199,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_199 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_201 = x_198 + x_200
        x_198 = x_200 = None
        float_53 = x_201.float()
        pow_53 = float_53.pow(2)
        mean_53 = pow_53.mean(-1, keepdim=True)
        pow_53 = None
        add_105 = mean_53 + 1e-05
        mean_53 = None
        rsqrt_52 = torch.rsqrt(add_105)
        add_105 = None
        mul_129 = float_53 * rsqrt_52
        float_53 = rsqrt_52 = None
        output_52 = mul_129.type_as(x_201)
        mul_129 = None
        mul_130 = (
            output_52
            * l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_1_modules_norm_1_parameters_weight_
        )
        output_52 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_1_modules_norm_1_parameters_weight_ = (None)
        linear_129 = torch._C._nn.linear(
            mul_130,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_,
            None,
        )
        mul_130 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_ = (None)
        reshape_54 = linear_129.reshape(2, 7, 3, 6, 128)
        linear_129 = None
        qkv_25 = reshape_54.permute(2, 0, 3, 1, 4)
        reshape_54 = None
        unbind_25 = qkv_25.unbind(0)
        qkv_25 = None
        q_26 = unbind_25[0]
        k_26 = unbind_25[1]
        v_26 = unbind_25[2]
        unbind_25 = None
        mask_2 = torch.full(
            (7, 7), -3.4028234663852886e38, device=device(type="cuda", index=0)
        )
        mask_cond_1 = torch.arange(7, device=device(type="cuda", index=0))
        add_106 = mask_cond_1 + 1
        view_1 = add_106.view(7, 1)
        add_106 = None
        lt_1 = mask_cond_1 < view_1
        mask_cond_1 = view_1 = None
        masked_fill__1 = mask_2.masked_fill_(lt_1, 0)
        lt_1 = masked_fill__1 = None
        mask_3 = mask_2.to(torch.float32)
        mask_2 = None
        getitem_83 = mask_3[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        mask_3 = None
        causal_4d_mask_1 = getitem_83.expand(2, 1, 7, 7)
        getitem_83 = None
        getitem_84 = l_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        expand_4 = getitem_84.expand(2, 1, 7, 7)
        getitem_84 = None
        expanded_mask_1 = expand_4.to(torch.float32)
        expand_4 = None
        tensor_1 = torch.tensor(1.0, dtype=torch.float32)
        inverted_mask_1 = tensor_1 - expanded_mask_1
        tensor_1 = expanded_mask_1 = None
        to_7 = inverted_mask_1.to(torch.bool)
        masked_fill_2 = inverted_mask_1.masked_fill(to_7, -3.4028234663852886e38)
        inverted_mask_1 = to_7 = None
        expanded_attn_mask_2 = masked_fill_2.to(device(type="cuda", index=0))
        masked_fill_2 = None
        bool_2 = expanded_attn_mask_2.bool()
        expanded_attn_mask_2 = None
        expanded_attn_mask_3 = causal_4d_mask_1.masked_fill(
            bool_2, -3.4028234663852886e38
        )
        causal_4d_mask_1 = bool_2 = None
        x_202 = torch._C._nn.scaled_dot_product_attention(
            q_26, k_26, v_26, attn_mask=expanded_attn_mask_3
        )
        q_26 = k_26 = v_26 = expanded_attn_mask_3 = None
        transpose_27 = x_202.transpose(1, 2)
        x_202 = None
        contiguous_25 = transpose_27.contiguous()
        transpose_27 = None
        x_203 = contiguous_25.reshape(2, 7, 768)
        contiguous_25 = None
        x_204 = torch._C._nn.linear(
            x_203,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_,
            None,
        )
        x_203 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_ = (None)
        x_205 = torch.nn.functional.dropout(x_204, 0.0, False, False)
        x_204 = None
        x_206 = x_201 + x_205
        x_201 = x_205 = None
        float_54 = x_206.float()
        pow_54 = float_54.pow(2)
        mean_54 = pow_54.mean(-1, keepdim=True)
        pow_54 = None
        add_108 = mean_54 + 1e-05
        mean_54 = None
        rsqrt_53 = torch.rsqrt(add_108)
        add_108 = None
        mul_131 = float_54 * rsqrt_53
        float_54 = rsqrt_53 = None
        output_53 = mul_131.type_as(x_206)
        mul_131 = None
        mul_132 = (
            output_53
            * l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_1_modules_norm_2_parameters_weight_
        )
        output_53 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_1_modules_norm_2_parameters_weight_ = (None)
        linear_131 = torch._C._nn.linear(
            mul_132,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = (
            None
        )
        silu_25 = torch.nn.functional.silu(linear_131)
        linear_131 = None
        linear_132 = torch._C._nn.linear(
            mul_132,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_1_modules_mlp_modules_fc3_parameters_weight_,
            None,
        )
        mul_132 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_1_modules_mlp_modules_fc3_parameters_weight_ = (None)
        x_207 = silu_25 * linear_132
        silu_25 = linear_132 = None
        x_208 = torch._C._nn.linear(
            x_207,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_207 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_209 = x_206 + x_208
        x_206 = x_208 = None
        float_55 = x_209.float()
        pow_55 = float_55.pow(2)
        mean_55 = pow_55.mean(-1, keepdim=True)
        pow_55 = None
        add_110 = mean_55 + 1e-05
        mean_55 = None
        rsqrt_54 = torch.rsqrt(add_110)
        add_110 = None
        mul_134 = float_55 * rsqrt_54
        float_55 = rsqrt_54 = None
        output_54 = mul_134.type_as(x_209)
        mul_134 = None
        mul_135 = (
            output_54
            * l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_2_modules_norm_1_parameters_weight_
        )
        output_54 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_2_modules_norm_1_parameters_weight_ = (None)
        linear_134 = torch._C._nn.linear(
            mul_135,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_,
            None,
        )
        mul_135 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_ = (None)
        reshape_56 = linear_134.reshape(2, 7, 3, 6, 128)
        linear_134 = None
        qkv_26 = reshape_56.permute(2, 0, 3, 1, 4)
        reshape_56 = None
        unbind_26 = qkv_26.unbind(0)
        qkv_26 = None
        q_27 = unbind_26[0]
        k_27 = unbind_26[1]
        v_27 = unbind_26[2]
        unbind_26 = None
        mask_4 = torch.full(
            (7, 7), -3.4028234663852886e38, device=device(type="cuda", index=0)
        )
        mask_cond_2 = torch.arange(7, device=device(type="cuda", index=0))
        add_111 = mask_cond_2 + 1
        view_2 = add_111.view(7, 1)
        add_111 = None
        lt_2 = mask_cond_2 < view_2
        mask_cond_2 = view_2 = None
        masked_fill__2 = mask_4.masked_fill_(lt_2, 0)
        lt_2 = masked_fill__2 = None
        mask_5 = mask_4.to(torch.float32)
        mask_4 = None
        getitem_88 = mask_5[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        mask_5 = None
        causal_4d_mask_2 = getitem_88.expand(2, 1, 7, 7)
        getitem_88 = None
        getitem_89 = l_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        expand_6 = getitem_89.expand(2, 1, 7, 7)
        getitem_89 = None
        expanded_mask_2 = expand_6.to(torch.float32)
        expand_6 = None
        tensor_2 = torch.tensor(1.0, dtype=torch.float32)
        inverted_mask_2 = tensor_2 - expanded_mask_2
        tensor_2 = expanded_mask_2 = None
        to_11 = inverted_mask_2.to(torch.bool)
        masked_fill_4 = inverted_mask_2.masked_fill(to_11, -3.4028234663852886e38)
        inverted_mask_2 = to_11 = None
        expanded_attn_mask_4 = masked_fill_4.to(device(type="cuda", index=0))
        masked_fill_4 = None
        bool_3 = expanded_attn_mask_4.bool()
        expanded_attn_mask_4 = None
        expanded_attn_mask_5 = causal_4d_mask_2.masked_fill(
            bool_3, -3.4028234663852886e38
        )
        causal_4d_mask_2 = bool_3 = None
        x_210 = torch._C._nn.scaled_dot_product_attention(
            q_27, k_27, v_27, attn_mask=expanded_attn_mask_5
        )
        q_27 = k_27 = v_27 = expanded_attn_mask_5 = None
        transpose_28 = x_210.transpose(1, 2)
        x_210 = None
        contiguous_26 = transpose_28.contiguous()
        transpose_28 = None
        x_211 = contiguous_26.reshape(2, 7, 768)
        contiguous_26 = None
        x_212 = torch._C._nn.linear(
            x_211,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_,
            None,
        )
        x_211 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_ = (None)
        x_213 = torch.nn.functional.dropout(x_212, 0.0, False, False)
        x_212 = None
        x_214 = x_209 + x_213
        x_209 = x_213 = None
        float_56 = x_214.float()
        pow_56 = float_56.pow(2)
        mean_56 = pow_56.mean(-1, keepdim=True)
        pow_56 = None
        add_113 = mean_56 + 1e-05
        mean_56 = None
        rsqrt_55 = torch.rsqrt(add_113)
        add_113 = None
        mul_136 = float_56 * rsqrt_55
        float_56 = rsqrt_55 = None
        output_55 = mul_136.type_as(x_214)
        mul_136 = None
        mul_137 = (
            output_55
            * l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_2_modules_norm_2_parameters_weight_
        )
        output_55 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_2_modules_norm_2_parameters_weight_ = (None)
        linear_136 = torch._C._nn.linear(
            mul_137,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = (
            None
        )
        silu_26 = torch.nn.functional.silu(linear_136)
        linear_136 = None
        linear_137 = torch._C._nn.linear(
            mul_137,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_2_modules_mlp_modules_fc3_parameters_weight_,
            None,
        )
        mul_137 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_2_modules_mlp_modules_fc3_parameters_weight_ = (None)
        x_215 = silu_26 * linear_137
        silu_26 = linear_137 = None
        x_216 = torch._C._nn.linear(
            x_215,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_215 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_217 = x_214 + x_216
        x_214 = x_216 = None
        float_57 = x_217.float()
        pow_57 = float_57.pow(2)
        mean_57 = pow_57.mean(-1, keepdim=True)
        pow_57 = None
        add_115 = mean_57 + 1e-05
        mean_57 = None
        rsqrt_56 = torch.rsqrt(add_115)
        add_115 = None
        mul_139 = float_57 * rsqrt_56
        float_57 = rsqrt_56 = None
        output_56 = mul_139.type_as(x_217)
        mul_139 = None
        mul_140 = (
            output_56
            * l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_3_modules_norm_1_parameters_weight_
        )
        output_56 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_3_modules_norm_1_parameters_weight_ = (None)
        linear_139 = torch._C._nn.linear(
            mul_140,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_,
            None,
        )
        mul_140 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_ = (None)
        reshape_58 = linear_139.reshape(2, 7, 3, 6, 128)
        linear_139 = None
        qkv_27 = reshape_58.permute(2, 0, 3, 1, 4)
        reshape_58 = None
        unbind_27 = qkv_27.unbind(0)
        qkv_27 = None
        q_28 = unbind_27[0]
        k_28 = unbind_27[1]
        v_28 = unbind_27[2]
        unbind_27 = None
        mask_6 = torch.full(
            (7, 7), -3.4028234663852886e38, device=device(type="cuda", index=0)
        )
        mask_cond_3 = torch.arange(7, device=device(type="cuda", index=0))
        add_116 = mask_cond_3 + 1
        view_3 = add_116.view(7, 1)
        add_116 = None
        lt_3 = mask_cond_3 < view_3
        mask_cond_3 = view_3 = None
        masked_fill__3 = mask_6.masked_fill_(lt_3, 0)
        lt_3 = masked_fill__3 = None
        mask_7 = mask_6.to(torch.float32)
        mask_6 = None
        getitem_93 = mask_7[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        mask_7 = None
        causal_4d_mask_3 = getitem_93.expand(2, 1, 7, 7)
        getitem_93 = None
        getitem_94 = l_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        expand_8 = getitem_94.expand(2, 1, 7, 7)
        getitem_94 = None
        expanded_mask_3 = expand_8.to(torch.float32)
        expand_8 = None
        tensor_3 = torch.tensor(1.0, dtype=torch.float32)
        inverted_mask_3 = tensor_3 - expanded_mask_3
        tensor_3 = expanded_mask_3 = None
        to_15 = inverted_mask_3.to(torch.bool)
        masked_fill_6 = inverted_mask_3.masked_fill(to_15, -3.4028234663852886e38)
        inverted_mask_3 = to_15 = None
        expanded_attn_mask_6 = masked_fill_6.to(device(type="cuda", index=0))
        masked_fill_6 = None
        bool_4 = expanded_attn_mask_6.bool()
        expanded_attn_mask_6 = None
        expanded_attn_mask_7 = causal_4d_mask_3.masked_fill(
            bool_4, -3.4028234663852886e38
        )
        causal_4d_mask_3 = bool_4 = None
        x_218 = torch._C._nn.scaled_dot_product_attention(
            q_28, k_28, v_28, attn_mask=expanded_attn_mask_7
        )
        q_28 = k_28 = v_28 = expanded_attn_mask_7 = None
        transpose_29 = x_218.transpose(1, 2)
        x_218 = None
        contiguous_27 = transpose_29.contiguous()
        transpose_29 = None
        x_219 = contiguous_27.reshape(2, 7, 768)
        contiguous_27 = None
        x_220 = torch._C._nn.linear(
            x_219,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_,
            None,
        )
        x_219 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_ = (None)
        x_221 = torch.nn.functional.dropout(x_220, 0.0, False, False)
        x_220 = None
        x_222 = x_217 + x_221
        x_217 = x_221 = None
        float_58 = x_222.float()
        pow_58 = float_58.pow(2)
        mean_58 = pow_58.mean(-1, keepdim=True)
        pow_58 = None
        add_118 = mean_58 + 1e-05
        mean_58 = None
        rsqrt_57 = torch.rsqrt(add_118)
        add_118 = None
        mul_141 = float_58 * rsqrt_57
        float_58 = rsqrt_57 = None
        output_57 = mul_141.type_as(x_222)
        mul_141 = None
        mul_142 = (
            output_57
            * l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_3_modules_norm_2_parameters_weight_
        )
        output_57 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_3_modules_norm_2_parameters_weight_ = (None)
        linear_141 = torch._C._nn.linear(
            mul_142,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = (
            None
        )
        silu_27 = torch.nn.functional.silu(linear_141)
        linear_141 = None
        linear_142 = torch._C._nn.linear(
            mul_142,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_3_modules_mlp_modules_fc3_parameters_weight_,
            None,
        )
        mul_142 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_3_modules_mlp_modules_fc3_parameters_weight_ = (None)
        x_223 = silu_27 * linear_142
        silu_27 = linear_142 = None
        x_224 = torch._C._nn.linear(
            x_223,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_223 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_225 = x_222 + x_224
        x_222 = x_224 = None
        float_59 = x_225.float()
        pow_59 = float_59.pow(2)
        mean_59 = pow_59.mean(-1, keepdim=True)
        pow_59 = None
        add_120 = mean_59 + 1e-05
        mean_59 = None
        rsqrt_58 = torch.rsqrt(add_120)
        add_120 = None
        mul_144 = float_59 * rsqrt_58
        float_59 = rsqrt_58 = None
        output_58 = mul_144.type_as(x_225)
        mul_144 = None
        mul_145 = (
            output_58
            * l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_4_modules_norm_1_parameters_weight_
        )
        output_58 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_4_modules_norm_1_parameters_weight_ = (None)
        linear_144 = torch._C._nn.linear(
            mul_145,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_,
            None,
        )
        mul_145 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_ = (None)
        reshape_60 = linear_144.reshape(2, 7, 3, 6, 128)
        linear_144 = None
        qkv_28 = reshape_60.permute(2, 0, 3, 1, 4)
        reshape_60 = None
        unbind_28 = qkv_28.unbind(0)
        qkv_28 = None
        q_29 = unbind_28[0]
        k_29 = unbind_28[1]
        v_29 = unbind_28[2]
        unbind_28 = None
        mask_8 = torch.full(
            (7, 7), -3.4028234663852886e38, device=device(type="cuda", index=0)
        )
        mask_cond_4 = torch.arange(7, device=device(type="cuda", index=0))
        add_121 = mask_cond_4 + 1
        view_4 = add_121.view(7, 1)
        add_121 = None
        lt_4 = mask_cond_4 < view_4
        mask_cond_4 = view_4 = None
        masked_fill__4 = mask_8.masked_fill_(lt_4, 0)
        lt_4 = masked_fill__4 = None
        mask_9 = mask_8.to(torch.float32)
        mask_8 = None
        getitem_98 = mask_9[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        mask_9 = None
        causal_4d_mask_4 = getitem_98.expand(2, 1, 7, 7)
        getitem_98 = None
        getitem_99 = l_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        expand_10 = getitem_99.expand(2, 1, 7, 7)
        getitem_99 = None
        expanded_mask_4 = expand_10.to(torch.float32)
        expand_10 = None
        tensor_4 = torch.tensor(1.0, dtype=torch.float32)
        inverted_mask_4 = tensor_4 - expanded_mask_4
        tensor_4 = expanded_mask_4 = None
        to_19 = inverted_mask_4.to(torch.bool)
        masked_fill_8 = inverted_mask_4.masked_fill(to_19, -3.4028234663852886e38)
        inverted_mask_4 = to_19 = None
        expanded_attn_mask_8 = masked_fill_8.to(device(type="cuda", index=0))
        masked_fill_8 = None
        bool_5 = expanded_attn_mask_8.bool()
        expanded_attn_mask_8 = None
        expanded_attn_mask_9 = causal_4d_mask_4.masked_fill(
            bool_5, -3.4028234663852886e38
        )
        causal_4d_mask_4 = bool_5 = None
        x_226 = torch._C._nn.scaled_dot_product_attention(
            q_29, k_29, v_29, attn_mask=expanded_attn_mask_9
        )
        q_29 = k_29 = v_29 = expanded_attn_mask_9 = None
        transpose_30 = x_226.transpose(1, 2)
        x_226 = None
        contiguous_28 = transpose_30.contiguous()
        transpose_30 = None
        x_227 = contiguous_28.reshape(2, 7, 768)
        contiguous_28 = None
        x_228 = torch._C._nn.linear(
            x_227,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_,
            None,
        )
        x_227 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_ = (None)
        x_229 = torch.nn.functional.dropout(x_228, 0.0, False, False)
        x_228 = None
        x_230 = x_225 + x_229
        x_225 = x_229 = None
        float_60 = x_230.float()
        pow_60 = float_60.pow(2)
        mean_60 = pow_60.mean(-1, keepdim=True)
        pow_60 = None
        add_123 = mean_60 + 1e-05
        mean_60 = None
        rsqrt_59 = torch.rsqrt(add_123)
        add_123 = None
        mul_146 = float_60 * rsqrt_59
        float_60 = rsqrt_59 = None
        output_59 = mul_146.type_as(x_230)
        mul_146 = None
        mul_147 = (
            output_59
            * l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_4_modules_norm_2_parameters_weight_
        )
        output_59 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_4_modules_norm_2_parameters_weight_ = (None)
        linear_146 = torch._C._nn.linear(
            mul_147,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = (
            None
        )
        silu_28 = torch.nn.functional.silu(linear_146)
        linear_146 = None
        linear_147 = torch._C._nn.linear(
            mul_147,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_4_modules_mlp_modules_fc3_parameters_weight_,
            None,
        )
        mul_147 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_4_modules_mlp_modules_fc3_parameters_weight_ = (None)
        x_231 = silu_28 * linear_147
        silu_28 = linear_147 = None
        x_232 = torch._C._nn.linear(
            x_231,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_231 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_233 = x_230 + x_232
        x_230 = x_232 = None
        float_61 = x_233.float()
        pow_61 = float_61.pow(2)
        mean_61 = pow_61.mean(-1, keepdim=True)
        pow_61 = None
        add_125 = mean_61 + 1e-05
        mean_61 = None
        rsqrt_60 = torch.rsqrt(add_125)
        add_125 = None
        mul_149 = float_61 * rsqrt_60
        float_61 = rsqrt_60 = None
        output_60 = mul_149.type_as(x_233)
        mul_149 = None
        mul_150 = (
            output_60
            * l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_5_modules_norm_1_parameters_weight_
        )
        output_60 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_5_modules_norm_1_parameters_weight_ = (None)
        linear_149 = torch._C._nn.linear(
            mul_150,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_,
            None,
        )
        mul_150 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_ = (None)
        reshape_62 = linear_149.reshape(2, 7, 3, 6, 128)
        linear_149 = None
        qkv_29 = reshape_62.permute(2, 0, 3, 1, 4)
        reshape_62 = None
        unbind_29 = qkv_29.unbind(0)
        qkv_29 = None
        q_30 = unbind_29[0]
        k_30 = unbind_29[1]
        v_30 = unbind_29[2]
        unbind_29 = None
        mask_10 = torch.full(
            (7, 7), -3.4028234663852886e38, device=device(type="cuda", index=0)
        )
        mask_cond_5 = torch.arange(7, device=device(type="cuda", index=0))
        add_126 = mask_cond_5 + 1
        view_5 = add_126.view(7, 1)
        add_126 = None
        lt_5 = mask_cond_5 < view_5
        mask_cond_5 = view_5 = None
        masked_fill__5 = mask_10.masked_fill_(lt_5, 0)
        lt_5 = masked_fill__5 = None
        mask_11 = mask_10.to(torch.float32)
        mask_10 = None
        getitem_103 = mask_11[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        mask_11 = None
        causal_4d_mask_5 = getitem_103.expand(2, 1, 7, 7)
        getitem_103 = None
        getitem_104 = l_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        expand_12 = getitem_104.expand(2, 1, 7, 7)
        getitem_104 = None
        expanded_mask_5 = expand_12.to(torch.float32)
        expand_12 = None
        tensor_5 = torch.tensor(1.0, dtype=torch.float32)
        inverted_mask_5 = tensor_5 - expanded_mask_5
        tensor_5 = expanded_mask_5 = None
        to_23 = inverted_mask_5.to(torch.bool)
        masked_fill_10 = inverted_mask_5.masked_fill(to_23, -3.4028234663852886e38)
        inverted_mask_5 = to_23 = None
        expanded_attn_mask_10 = masked_fill_10.to(device(type="cuda", index=0))
        masked_fill_10 = None
        bool_6 = expanded_attn_mask_10.bool()
        expanded_attn_mask_10 = None
        expanded_attn_mask_11 = causal_4d_mask_5.masked_fill(
            bool_6, -3.4028234663852886e38
        )
        causal_4d_mask_5 = bool_6 = None
        x_234 = torch._C._nn.scaled_dot_product_attention(
            q_30, k_30, v_30, attn_mask=expanded_attn_mask_11
        )
        q_30 = k_30 = v_30 = expanded_attn_mask_11 = None
        transpose_31 = x_234.transpose(1, 2)
        x_234 = None
        contiguous_29 = transpose_31.contiguous()
        transpose_31 = None
        x_235 = contiguous_29.reshape(2, 7, 768)
        contiguous_29 = None
        x_236 = torch._C._nn.linear(
            x_235,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_,
            None,
        )
        x_235 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_ = (None)
        x_237 = torch.nn.functional.dropout(x_236, 0.0, False, False)
        x_236 = None
        x_238 = x_233 + x_237
        x_233 = x_237 = None
        float_62 = x_238.float()
        pow_62 = float_62.pow(2)
        mean_62 = pow_62.mean(-1, keepdim=True)
        pow_62 = None
        add_128 = mean_62 + 1e-05
        mean_62 = None
        rsqrt_61 = torch.rsqrt(add_128)
        add_128 = None
        mul_151 = float_62 * rsqrt_61
        float_62 = rsqrt_61 = None
        output_61 = mul_151.type_as(x_238)
        mul_151 = None
        mul_152 = (
            output_61
            * l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_5_modules_norm_2_parameters_weight_
        )
        output_61 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_5_modules_norm_2_parameters_weight_ = (None)
        linear_151 = torch._C._nn.linear(
            mul_152,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = (
            None
        )
        silu_29 = torch.nn.functional.silu(linear_151)
        linear_151 = None
        linear_152 = torch._C._nn.linear(
            mul_152,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_5_modules_mlp_modules_fc3_parameters_weight_,
            None,
        )
        mul_152 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_5_modules_mlp_modules_fc3_parameters_weight_ = (None)
        x_239 = silu_29 * linear_152
        silu_29 = linear_152 = None
        x_240 = torch._C._nn.linear(
            x_239,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_239 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_241 = x_238 + x_240
        x_238 = x_240 = None
        float_63 = x_241.float()
        pow_63 = float_63.pow(2)
        mean_63 = pow_63.mean(-1, keepdim=True)
        pow_63 = None
        add_130 = mean_63 + 1e-05
        mean_63 = None
        rsqrt_62 = torch.rsqrt(add_130)
        add_130 = None
        mul_154 = float_63 * rsqrt_62
        float_63 = rsqrt_62 = None
        output_62 = mul_154.type_as(x_241)
        mul_154 = None
        mul_155 = (
            output_62
            * l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_6_modules_norm_1_parameters_weight_
        )
        output_62 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_6_modules_norm_1_parameters_weight_ = (None)
        linear_154 = torch._C._nn.linear(
            mul_155,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_,
            None,
        )
        mul_155 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_ = (None)
        reshape_64 = linear_154.reshape(2, 7, 3, 6, 128)
        linear_154 = None
        qkv_30 = reshape_64.permute(2, 0, 3, 1, 4)
        reshape_64 = None
        unbind_30 = qkv_30.unbind(0)
        qkv_30 = None
        q_31 = unbind_30[0]
        k_31 = unbind_30[1]
        v_31 = unbind_30[2]
        unbind_30 = None
        mask_12 = torch.full(
            (7, 7), -3.4028234663852886e38, device=device(type="cuda", index=0)
        )
        mask_cond_6 = torch.arange(7, device=device(type="cuda", index=0))
        add_131 = mask_cond_6 + 1
        view_6 = add_131.view(7, 1)
        add_131 = None
        lt_6 = mask_cond_6 < view_6
        mask_cond_6 = view_6 = None
        masked_fill__6 = mask_12.masked_fill_(lt_6, 0)
        lt_6 = masked_fill__6 = None
        mask_13 = mask_12.to(torch.float32)
        mask_12 = None
        getitem_108 = mask_13[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        mask_13 = None
        causal_4d_mask_6 = getitem_108.expand(2, 1, 7, 7)
        getitem_108 = None
        getitem_109 = l_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        expand_14 = getitem_109.expand(2, 1, 7, 7)
        getitem_109 = None
        expanded_mask_6 = expand_14.to(torch.float32)
        expand_14 = None
        tensor_6 = torch.tensor(1.0, dtype=torch.float32)
        inverted_mask_6 = tensor_6 - expanded_mask_6
        tensor_6 = expanded_mask_6 = None
        to_27 = inverted_mask_6.to(torch.bool)
        masked_fill_12 = inverted_mask_6.masked_fill(to_27, -3.4028234663852886e38)
        inverted_mask_6 = to_27 = None
        expanded_attn_mask_12 = masked_fill_12.to(device(type="cuda", index=0))
        masked_fill_12 = None
        bool_7 = expanded_attn_mask_12.bool()
        expanded_attn_mask_12 = None
        expanded_attn_mask_13 = causal_4d_mask_6.masked_fill(
            bool_7, -3.4028234663852886e38
        )
        causal_4d_mask_6 = bool_7 = None
        x_242 = torch._C._nn.scaled_dot_product_attention(
            q_31, k_31, v_31, attn_mask=expanded_attn_mask_13
        )
        q_31 = k_31 = v_31 = expanded_attn_mask_13 = None
        transpose_32 = x_242.transpose(1, 2)
        x_242 = None
        contiguous_30 = transpose_32.contiguous()
        transpose_32 = None
        x_243 = contiguous_30.reshape(2, 7, 768)
        contiguous_30 = None
        x_244 = torch._C._nn.linear(
            x_243,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_,
            None,
        )
        x_243 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_ = (None)
        x_245 = torch.nn.functional.dropout(x_244, 0.0, False, False)
        x_244 = None
        x_246 = x_241 + x_245
        x_241 = x_245 = None
        float_64 = x_246.float()
        pow_64 = float_64.pow(2)
        mean_64 = pow_64.mean(-1, keepdim=True)
        pow_64 = None
        add_133 = mean_64 + 1e-05
        mean_64 = None
        rsqrt_63 = torch.rsqrt(add_133)
        add_133 = None
        mul_156 = float_64 * rsqrt_63
        float_64 = rsqrt_63 = None
        output_63 = mul_156.type_as(x_246)
        mul_156 = None
        mul_157 = (
            output_63
            * l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_6_modules_norm_2_parameters_weight_
        )
        output_63 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_6_modules_norm_2_parameters_weight_ = (None)
        linear_156 = torch._C._nn.linear(
            mul_157,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = (
            None
        )
        silu_30 = torch.nn.functional.silu(linear_156)
        linear_156 = None
        linear_157 = torch._C._nn.linear(
            mul_157,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_6_modules_mlp_modules_fc3_parameters_weight_,
            None,
        )
        mul_157 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_6_modules_mlp_modules_fc3_parameters_weight_ = (None)
        x_247 = silu_30 * linear_157
        silu_30 = linear_157 = None
        x_248 = torch._C._nn.linear(
            x_247,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_247 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_249 = x_246 + x_248
        x_246 = x_248 = None
        float_65 = x_249.float()
        pow_65 = float_65.pow(2)
        mean_65 = pow_65.mean(-1, keepdim=True)
        pow_65 = None
        add_135 = mean_65 + 1e-05
        mean_65 = None
        rsqrt_64 = torch.rsqrt(add_135)
        add_135 = None
        mul_159 = float_65 * rsqrt_64
        float_65 = rsqrt_64 = None
        output_64 = mul_159.type_as(x_249)
        mul_159 = None
        mul_160 = (
            output_64
            * l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_7_modules_norm_1_parameters_weight_
        )
        output_64 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_7_modules_norm_1_parameters_weight_ = (None)
        linear_159 = torch._C._nn.linear(
            mul_160,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_,
            None,
        )
        mul_160 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_ = (None)
        reshape_66 = linear_159.reshape(2, 7, 3, 6, 128)
        linear_159 = None
        qkv_31 = reshape_66.permute(2, 0, 3, 1, 4)
        reshape_66 = None
        unbind_31 = qkv_31.unbind(0)
        qkv_31 = None
        q_32 = unbind_31[0]
        k_32 = unbind_31[1]
        v_32 = unbind_31[2]
        unbind_31 = None
        mask_14 = torch.full(
            (7, 7), -3.4028234663852886e38, device=device(type="cuda", index=0)
        )
        mask_cond_7 = torch.arange(7, device=device(type="cuda", index=0))
        add_136 = mask_cond_7 + 1
        view_7 = add_136.view(7, 1)
        add_136 = None
        lt_7 = mask_cond_7 < view_7
        mask_cond_7 = view_7 = None
        masked_fill__7 = mask_14.masked_fill_(lt_7, 0)
        lt_7 = masked_fill__7 = None
        mask_15 = mask_14.to(torch.float32)
        mask_14 = None
        getitem_113 = mask_15[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        mask_15 = None
        causal_4d_mask_7 = getitem_113.expand(2, 1, 7, 7)
        getitem_113 = None
        getitem_114 = l_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        expand_16 = getitem_114.expand(2, 1, 7, 7)
        getitem_114 = None
        expanded_mask_7 = expand_16.to(torch.float32)
        expand_16 = None
        tensor_7 = torch.tensor(1.0, dtype=torch.float32)
        inverted_mask_7 = tensor_7 - expanded_mask_7
        tensor_7 = expanded_mask_7 = None
        to_31 = inverted_mask_7.to(torch.bool)
        masked_fill_14 = inverted_mask_7.masked_fill(to_31, -3.4028234663852886e38)
        inverted_mask_7 = to_31 = None
        expanded_attn_mask_14 = masked_fill_14.to(device(type="cuda", index=0))
        masked_fill_14 = None
        bool_8 = expanded_attn_mask_14.bool()
        expanded_attn_mask_14 = None
        expanded_attn_mask_15 = causal_4d_mask_7.masked_fill(
            bool_8, -3.4028234663852886e38
        )
        causal_4d_mask_7 = bool_8 = None
        x_250 = torch._C._nn.scaled_dot_product_attention(
            q_32, k_32, v_32, attn_mask=expanded_attn_mask_15
        )
        q_32 = k_32 = v_32 = expanded_attn_mask_15 = None
        transpose_33 = x_250.transpose(1, 2)
        x_250 = None
        contiguous_31 = transpose_33.contiguous()
        transpose_33 = None
        x_251 = contiguous_31.reshape(2, 7, 768)
        contiguous_31 = None
        x_252 = torch._C._nn.linear(
            x_251,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_,
            None,
        )
        x_251 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_ = (None)
        x_253 = torch.nn.functional.dropout(x_252, 0.0, False, False)
        x_252 = None
        x_254 = x_249 + x_253
        x_249 = x_253 = None
        float_66 = x_254.float()
        pow_66 = float_66.pow(2)
        mean_66 = pow_66.mean(-1, keepdim=True)
        pow_66 = None
        add_138 = mean_66 + 1e-05
        mean_66 = None
        rsqrt_65 = torch.rsqrt(add_138)
        add_138 = None
        mul_161 = float_66 * rsqrt_65
        float_66 = rsqrt_65 = None
        output_65 = mul_161.type_as(x_254)
        mul_161 = None
        mul_162 = (
            output_65
            * l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_7_modules_norm_2_parameters_weight_
        )
        output_65 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_7_modules_norm_2_parameters_weight_ = (None)
        linear_161 = torch._C._nn.linear(
            mul_162,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = (
            None
        )
        silu_31 = torch.nn.functional.silu(linear_161)
        linear_161 = None
        linear_162 = torch._C._nn.linear(
            mul_162,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_7_modules_mlp_modules_fc3_parameters_weight_,
            None,
        )
        mul_162 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_7_modules_mlp_modules_fc3_parameters_weight_ = (None)
        x_255 = silu_31 * linear_162
        silu_31 = linear_162 = None
        x_256 = torch._C._nn.linear(
            x_255,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_255 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_257 = x_254 + x_256
        x_254 = x_256 = None
        float_67 = x_257.float()
        pow_67 = float_67.pow(2)
        mean_67 = pow_67.mean(-1, keepdim=True)
        pow_67 = None
        add_140 = mean_67 + 1e-05
        mean_67 = None
        rsqrt_66 = torch.rsqrt(add_140)
        add_140 = None
        mul_164 = float_67 * rsqrt_66
        float_67 = rsqrt_66 = None
        output_66 = mul_164.type_as(x_257)
        mul_164 = None
        mul_165 = (
            output_66
            * l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_8_modules_norm_1_parameters_weight_
        )
        output_66 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_8_modules_norm_1_parameters_weight_ = (None)
        linear_164 = torch._C._nn.linear(
            mul_165,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_,
            None,
        )
        mul_165 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_ = (None)
        reshape_68 = linear_164.reshape(2, 7, 3, 6, 128)
        linear_164 = None
        qkv_32 = reshape_68.permute(2, 0, 3, 1, 4)
        reshape_68 = None
        unbind_32 = qkv_32.unbind(0)
        qkv_32 = None
        q_33 = unbind_32[0]
        k_33 = unbind_32[1]
        v_33 = unbind_32[2]
        unbind_32 = None
        mask_16 = torch.full(
            (7, 7), -3.4028234663852886e38, device=device(type="cuda", index=0)
        )
        mask_cond_8 = torch.arange(7, device=device(type="cuda", index=0))
        add_141 = mask_cond_8 + 1
        view_8 = add_141.view(7, 1)
        add_141 = None
        lt_8 = mask_cond_8 < view_8
        mask_cond_8 = view_8 = None
        masked_fill__8 = mask_16.masked_fill_(lt_8, 0)
        lt_8 = masked_fill__8 = None
        mask_17 = mask_16.to(torch.float32)
        mask_16 = None
        getitem_118 = mask_17[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        mask_17 = None
        causal_4d_mask_8 = getitem_118.expand(2, 1, 7, 7)
        getitem_118 = None
        getitem_119 = l_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        expand_18 = getitem_119.expand(2, 1, 7, 7)
        getitem_119 = None
        expanded_mask_8 = expand_18.to(torch.float32)
        expand_18 = None
        tensor_8 = torch.tensor(1.0, dtype=torch.float32)
        inverted_mask_8 = tensor_8 - expanded_mask_8
        tensor_8 = expanded_mask_8 = None
        to_35 = inverted_mask_8.to(torch.bool)
        masked_fill_16 = inverted_mask_8.masked_fill(to_35, -3.4028234663852886e38)
        inverted_mask_8 = to_35 = None
        expanded_attn_mask_16 = masked_fill_16.to(device(type="cuda", index=0))
        masked_fill_16 = None
        bool_9 = expanded_attn_mask_16.bool()
        expanded_attn_mask_16 = None
        expanded_attn_mask_17 = causal_4d_mask_8.masked_fill(
            bool_9, -3.4028234663852886e38
        )
        causal_4d_mask_8 = bool_9 = None
        x_258 = torch._C._nn.scaled_dot_product_attention(
            q_33, k_33, v_33, attn_mask=expanded_attn_mask_17
        )
        q_33 = k_33 = v_33 = expanded_attn_mask_17 = None
        transpose_34 = x_258.transpose(1, 2)
        x_258 = None
        contiguous_32 = transpose_34.contiguous()
        transpose_34 = None
        x_259 = contiguous_32.reshape(2, 7, 768)
        contiguous_32 = None
        x_260 = torch._C._nn.linear(
            x_259,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_,
            None,
        )
        x_259 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_ = (None)
        x_261 = torch.nn.functional.dropout(x_260, 0.0, False, False)
        x_260 = None
        x_262 = x_257 + x_261
        x_257 = x_261 = None
        float_68 = x_262.float()
        pow_68 = float_68.pow(2)
        mean_68 = pow_68.mean(-1, keepdim=True)
        pow_68 = None
        add_143 = mean_68 + 1e-05
        mean_68 = None
        rsqrt_67 = torch.rsqrt(add_143)
        add_143 = None
        mul_166 = float_68 * rsqrt_67
        float_68 = rsqrt_67 = None
        output_67 = mul_166.type_as(x_262)
        mul_166 = None
        mul_167 = (
            output_67
            * l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_8_modules_norm_2_parameters_weight_
        )
        output_67 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_8_modules_norm_2_parameters_weight_ = (None)
        linear_166 = torch._C._nn.linear(
            mul_167,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_ = (
            None
        )
        silu_32 = torch.nn.functional.silu(linear_166)
        linear_166 = None
        linear_167 = torch._C._nn.linear(
            mul_167,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_8_modules_mlp_modules_fc3_parameters_weight_,
            None,
        )
        mul_167 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_8_modules_mlp_modules_fc3_parameters_weight_ = (None)
        x_263 = silu_32 * linear_167
        silu_32 = linear_167 = None
        x_264 = torch._C._nn.linear(
            x_263,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_263 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_265 = x_262 + x_264
        x_262 = x_264 = None
        float_69 = x_265.float()
        pow_69 = float_69.pow(2)
        mean_69 = pow_69.mean(-1, keepdim=True)
        pow_69 = None
        add_145 = mean_69 + 1e-05
        mean_69 = None
        rsqrt_68 = torch.rsqrt(add_145)
        add_145 = None
        mul_169 = float_69 * rsqrt_68
        float_69 = rsqrt_68 = None
        output_68 = mul_169.type_as(x_265)
        mul_169 = None
        mul_170 = (
            output_68
            * l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_9_modules_norm_1_parameters_weight_
        )
        output_68 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_9_modules_norm_1_parameters_weight_ = (None)
        linear_169 = torch._C._nn.linear(
            mul_170,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_,
            None,
        )
        mul_170 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_ = (None)
        reshape_70 = linear_169.reshape(2, 7, 3, 6, 128)
        linear_169 = None
        qkv_33 = reshape_70.permute(2, 0, 3, 1, 4)
        reshape_70 = None
        unbind_33 = qkv_33.unbind(0)
        qkv_33 = None
        q_34 = unbind_33[0]
        k_34 = unbind_33[1]
        v_34 = unbind_33[2]
        unbind_33 = None
        mask_18 = torch.full(
            (7, 7), -3.4028234663852886e38, device=device(type="cuda", index=0)
        )
        mask_cond_9 = torch.arange(7, device=device(type="cuda", index=0))
        add_146 = mask_cond_9 + 1
        view_9 = add_146.view(7, 1)
        add_146 = None
        lt_9 = mask_cond_9 < view_9
        mask_cond_9 = view_9 = None
        masked_fill__9 = mask_18.masked_fill_(lt_9, 0)
        lt_9 = masked_fill__9 = None
        mask_19 = mask_18.to(torch.float32)
        mask_18 = None
        getitem_123 = mask_19[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        mask_19 = None
        causal_4d_mask_9 = getitem_123.expand(2, 1, 7, 7)
        getitem_123 = None
        getitem_124 = l_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        expand_20 = getitem_124.expand(2, 1, 7, 7)
        getitem_124 = None
        expanded_mask_9 = expand_20.to(torch.float32)
        expand_20 = None
        tensor_9 = torch.tensor(1.0, dtype=torch.float32)
        inverted_mask_9 = tensor_9 - expanded_mask_9
        tensor_9 = expanded_mask_9 = None
        to_39 = inverted_mask_9.to(torch.bool)
        masked_fill_18 = inverted_mask_9.masked_fill(to_39, -3.4028234663852886e38)
        inverted_mask_9 = to_39 = None
        expanded_attn_mask_18 = masked_fill_18.to(device(type="cuda", index=0))
        masked_fill_18 = None
        bool_10 = expanded_attn_mask_18.bool()
        expanded_attn_mask_18 = None
        expanded_attn_mask_19 = causal_4d_mask_9.masked_fill(
            bool_10, -3.4028234663852886e38
        )
        causal_4d_mask_9 = bool_10 = None
        x_266 = torch._C._nn.scaled_dot_product_attention(
            q_34, k_34, v_34, attn_mask=expanded_attn_mask_19
        )
        q_34 = k_34 = v_34 = expanded_attn_mask_19 = None
        transpose_35 = x_266.transpose(1, 2)
        x_266 = None
        contiguous_33 = transpose_35.contiguous()
        transpose_35 = None
        x_267 = contiguous_33.reshape(2, 7, 768)
        contiguous_33 = None
        x_268 = torch._C._nn.linear(
            x_267,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_,
            None,
        )
        x_267 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_ = (None)
        x_269 = torch.nn.functional.dropout(x_268, 0.0, False, False)
        x_268 = None
        x_270 = x_265 + x_269
        x_265 = x_269 = None
        float_70 = x_270.float()
        pow_70 = float_70.pow(2)
        mean_70 = pow_70.mean(-1, keepdim=True)
        pow_70 = None
        add_148 = mean_70 + 1e-05
        mean_70 = None
        rsqrt_69 = torch.rsqrt(add_148)
        add_148 = None
        mul_171 = float_70 * rsqrt_69
        float_70 = rsqrt_69 = None
        output_69 = mul_171.type_as(x_270)
        mul_171 = None
        mul_172 = (
            output_69
            * l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_9_modules_norm_2_parameters_weight_
        )
        output_69 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_9_modules_norm_2_parameters_weight_ = (None)
        linear_171 = torch._C._nn.linear(
            mul_172,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_ = (
            None
        )
        silu_33 = torch.nn.functional.silu(linear_171)
        linear_171 = None
        linear_172 = torch._C._nn.linear(
            mul_172,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_9_modules_mlp_modules_fc3_parameters_weight_,
            None,
        )
        mul_172 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_9_modules_mlp_modules_fc3_parameters_weight_ = (None)
        x_271 = silu_33 * linear_172
        silu_33 = linear_172 = None
        x_272 = torch._C._nn.linear(
            x_271,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_271 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_273 = x_270 + x_272
        x_270 = x_272 = None
        float_71 = x_273.float()
        pow_71 = float_71.pow(2)
        mean_71 = pow_71.mean(-1, keepdim=True)
        pow_71 = None
        add_150 = mean_71 + 1e-05
        mean_71 = None
        rsqrt_70 = torch.rsqrt(add_150)
        add_150 = None
        mul_174 = float_71 * rsqrt_70
        float_71 = rsqrt_70 = None
        output_70 = mul_174.type_as(x_273)
        mul_174 = None
        mul_175 = (
            output_70
            * l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_10_modules_norm_1_parameters_weight_
        )
        output_70 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_10_modules_norm_1_parameters_weight_ = (None)
        linear_174 = torch._C._nn.linear(
            mul_175,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_,
            None,
        )
        mul_175 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_ = (None)
        reshape_72 = linear_174.reshape(2, 7, 3, 6, 128)
        linear_174 = None
        qkv_34 = reshape_72.permute(2, 0, 3, 1, 4)
        reshape_72 = None
        unbind_34 = qkv_34.unbind(0)
        qkv_34 = None
        q_35 = unbind_34[0]
        k_35 = unbind_34[1]
        v_35 = unbind_34[2]
        unbind_34 = None
        mask_20 = torch.full(
            (7, 7), -3.4028234663852886e38, device=device(type="cuda", index=0)
        )
        mask_cond_10 = torch.arange(7, device=device(type="cuda", index=0))
        add_151 = mask_cond_10 + 1
        view_10 = add_151.view(7, 1)
        add_151 = None
        lt_10 = mask_cond_10 < view_10
        mask_cond_10 = view_10 = None
        masked_fill__10 = mask_20.masked_fill_(lt_10, 0)
        lt_10 = masked_fill__10 = None
        mask_21 = mask_20.to(torch.float32)
        mask_20 = None
        getitem_128 = mask_21[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        mask_21 = None
        causal_4d_mask_10 = getitem_128.expand(2, 1, 7, 7)
        getitem_128 = None
        getitem_129 = l_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        expand_22 = getitem_129.expand(2, 1, 7, 7)
        getitem_129 = None
        expanded_mask_10 = expand_22.to(torch.float32)
        expand_22 = None
        tensor_10 = torch.tensor(1.0, dtype=torch.float32)
        inverted_mask_10 = tensor_10 - expanded_mask_10
        tensor_10 = expanded_mask_10 = None
        to_43 = inverted_mask_10.to(torch.bool)
        masked_fill_20 = inverted_mask_10.masked_fill(to_43, -3.4028234663852886e38)
        inverted_mask_10 = to_43 = None
        expanded_attn_mask_20 = masked_fill_20.to(device(type="cuda", index=0))
        masked_fill_20 = None
        bool_11 = expanded_attn_mask_20.bool()
        expanded_attn_mask_20 = None
        expanded_attn_mask_21 = causal_4d_mask_10.masked_fill(
            bool_11, -3.4028234663852886e38
        )
        causal_4d_mask_10 = bool_11 = None
        x_274 = torch._C._nn.scaled_dot_product_attention(
            q_35, k_35, v_35, attn_mask=expanded_attn_mask_21
        )
        q_35 = k_35 = v_35 = expanded_attn_mask_21 = None
        transpose_36 = x_274.transpose(1, 2)
        x_274 = None
        contiguous_34 = transpose_36.contiguous()
        transpose_36 = None
        x_275 = contiguous_34.reshape(2, 7, 768)
        contiguous_34 = None
        x_276 = torch._C._nn.linear(
            x_275,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_,
            None,
        )
        x_275 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_ = (None)
        x_277 = torch.nn.functional.dropout(x_276, 0.0, False, False)
        x_276 = None
        x_278 = x_273 + x_277
        x_273 = x_277 = None
        float_72 = x_278.float()
        pow_72 = float_72.pow(2)
        mean_72 = pow_72.mean(-1, keepdim=True)
        pow_72 = None
        add_153 = mean_72 + 1e-05
        mean_72 = None
        rsqrt_71 = torch.rsqrt(add_153)
        add_153 = None
        mul_176 = float_72 * rsqrt_71
        float_72 = rsqrt_71 = None
        output_71 = mul_176.type_as(x_278)
        mul_176 = None
        mul_177 = (
            output_71
            * l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_10_modules_norm_2_parameters_weight_
        )
        output_71 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_10_modules_norm_2_parameters_weight_ = (None)
        linear_176 = torch._C._nn.linear(
            mul_177,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_ = (
            None
        )
        silu_34 = torch.nn.functional.silu(linear_176)
        linear_176 = None
        linear_177 = torch._C._nn.linear(
            mul_177,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_10_modules_mlp_modules_fc3_parameters_weight_,
            None,
        )
        mul_177 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_10_modules_mlp_modules_fc3_parameters_weight_ = (None)
        x_279 = silu_34 * linear_177
        silu_34 = linear_177 = None
        x_280 = torch._C._nn.linear(
            x_279,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_279 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_281 = x_278 + x_280
        x_278 = x_280 = None
        float_73 = x_281.float()
        pow_73 = float_73.pow(2)
        mean_73 = pow_73.mean(-1, keepdim=True)
        pow_73 = None
        add_155 = mean_73 + 1e-05
        mean_73 = None
        rsqrt_72 = torch.rsqrt(add_155)
        add_155 = None
        mul_179 = float_73 * rsqrt_72
        float_73 = rsqrt_72 = None
        output_72 = mul_179.type_as(x_281)
        mul_179 = None
        mul_180 = (
            output_72
            * l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_11_modules_norm_1_parameters_weight_
        )
        output_72 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_11_modules_norm_1_parameters_weight_ = (None)
        linear_179 = torch._C._nn.linear(
            mul_180,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_,
            None,
        )
        mul_180 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_ = (None)
        reshape_74 = linear_179.reshape(2, 7, 3, 6, 128)
        linear_179 = None
        qkv_35 = reshape_74.permute(2, 0, 3, 1, 4)
        reshape_74 = None
        unbind_35 = qkv_35.unbind(0)
        qkv_35 = None
        q_36 = unbind_35[0]
        k_36 = unbind_35[1]
        v_36 = unbind_35[2]
        unbind_35 = None
        mask_22 = torch.full(
            (7, 7), -3.4028234663852886e38, device=device(type="cuda", index=0)
        )
        mask_cond_11 = torch.arange(7, device=device(type="cuda", index=0))
        add_156 = mask_cond_11 + 1
        view_11 = add_156.view(7, 1)
        add_156 = None
        lt_11 = mask_cond_11 < view_11
        mask_cond_11 = view_11 = None
        masked_fill__11 = mask_22.masked_fill_(lt_11, 0)
        lt_11 = masked_fill__11 = None
        mask_23 = mask_22.to(torch.float32)
        mask_22 = None
        getitem_133 = mask_23[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        mask_23 = None
        causal_4d_mask_11 = getitem_133.expand(2, 1, 7, 7)
        getitem_133 = None
        getitem_134 = l_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        l_attention_mask_ = None
        expand_24 = getitem_134.expand(2, 1, 7, 7)
        getitem_134 = None
        expanded_mask_11 = expand_24.to(torch.float32)
        expand_24 = None
        tensor_11 = torch.tensor(1.0, dtype=torch.float32)
        inverted_mask_11 = tensor_11 - expanded_mask_11
        tensor_11 = expanded_mask_11 = None
        to_47 = inverted_mask_11.to(torch.bool)
        masked_fill_22 = inverted_mask_11.masked_fill(to_47, -3.4028234663852886e38)
        inverted_mask_11 = to_47 = None
        expanded_attn_mask_22 = masked_fill_22.to(device(type="cuda", index=0))
        masked_fill_22 = None
        bool_12 = expanded_attn_mask_22.bool()
        expanded_attn_mask_22 = None
        expanded_attn_mask_23 = causal_4d_mask_11.masked_fill(
            bool_12, -3.4028234663852886e38
        )
        causal_4d_mask_11 = bool_12 = None
        x_282 = torch._C._nn.scaled_dot_product_attention(
            q_36, k_36, v_36, attn_mask=expanded_attn_mask_23
        )
        q_36 = k_36 = v_36 = expanded_attn_mask_23 = None
        transpose_37 = x_282.transpose(1, 2)
        x_282 = None
        contiguous_35 = transpose_37.contiguous()
        transpose_37 = None
        x_283 = contiguous_35.reshape(2, 7, 768)
        contiguous_35 = None
        x_284 = torch._C._nn.linear(
            x_283,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_,
            None,
        )
        x_283 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_ = (None)
        x_285 = torch.nn.functional.dropout(x_284, 0.0, False, False)
        x_284 = None
        x_286 = x_281 + x_285
        x_281 = x_285 = None
        float_74 = x_286.float()
        pow_74 = float_74.pow(2)
        mean_74 = pow_74.mean(-1, keepdim=True)
        pow_74 = None
        add_158 = mean_74 + 1e-05
        mean_74 = None
        rsqrt_73 = torch.rsqrt(add_158)
        add_158 = None
        mul_181 = float_74 * rsqrt_73
        float_74 = rsqrt_73 = None
        output_73 = mul_181.type_as(x_286)
        mul_181 = None
        mul_182 = (
            output_73
            * l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_11_modules_norm_2_parameters_weight_
        )
        output_73 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_11_modules_norm_2_parameters_weight_ = (None)
        linear_181 = torch._C._nn.linear(
            mul_182,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_,
            None,
        )
        l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_ = (
            None
        )
        silu_35 = torch.nn.functional.silu(linear_181)
        linear_181 = None
        linear_182 = torch._C._nn.linear(
            mul_182,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_11_modules_mlp_modules_fc3_parameters_weight_,
            None,
        )
        mul_182 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_11_modules_mlp_modules_fc3_parameters_weight_ = (None)
        x_287 = silu_35 * linear_182
        silu_35 = linear_182 = None
        x_288 = torch._C._nn.linear(
            x_287,
            l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_,
            None,
        )
        x_287 = l_self_modules_text_encoder_modules_trunk_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_289 = x_286 + x_288
        x_286 = x_288 = None
        float_75 = x_289.float()
        pow_75 = float_75.pow(2)
        mean_75 = pow_75.mean(-1, keepdim=True)
        pow_75 = None
        add_160 = mean_75 + 1e-05
        mean_75 = None
        rsqrt_74 = torch.rsqrt(add_160)
        add_160 = None
        mul_184 = float_75 * rsqrt_74
        float_75 = rsqrt_74 = None
        output_74 = mul_184.type_as(x_289)
        mul_184 = x_289 = None
        tokens_4 = (
            output_74
            * l_self_modules_text_encoder_modules_trunk_modules_post_trunk_norm_parameters_weight_
        )
        output_74 = l_self_modules_text_encoder_modules_trunk_modules_post_trunk_norm_parameters_weight_ = (None)
        float_76 = eos_token_mask.float()
        eos_token_mask = None
        eos_token_mask_1 = torch.argmax(float_76, dim=-1)
        float_76 = None
        reshape_76 = eos_token_mask_1.reshape(2, 1, 1)
        eos_token_mask_1 = None
        eos_token_mask_2 = reshape_76.expand(2, 1, 768)
        reshape_76 = None
        eos_token = torch.gather(tokens_4, 1, eos_token_mask_2)
        tokens_4 = eos_token_mask_2 = None
        eos_token_1 = eos_token.squeeze(1)
        eos_token = None
        text_features = torch._C._nn.linear(
            eos_token_1, l_self_modules_text_projector_parameters_weight_, None
        )
        l_self_modules_text_projector_parameters_weight_ = None
        text_features_1 = torch.nn.functional.normalize(text_features, p=2, dim=-1)
        text_features = None
        clamp = l_self_parameters_log_logit_scale_.clamp(0.0, 4.605170185988092)
        l_self_parameters_log_logit_scale_ = None
        logit_scale = clamp.exp()
        clamp = None
        mul_186 = logit_scale * text_features_1
        logit_scale = None
        t = image_features_1.t()
        logits_per_text = mul_186 @ t
        mul_186 = t = None
        logits_per_image = logits_per_text.t()
        return (
            out,
            eos_token_1,
            logits_per_image,
            logits_per_text,
            image_features_1,
            text_features_1,
        )
