import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_kwargs_pixel_values_: torch.Tensor,
        L_self_modules_shared_image_embedding_buffers_positional_embedding_: torch.Tensor,
        G_import_transformers_dot_models_dot_sam_dot_modeling_sam_np_pi: torch.Tensor,
        L_self_modules_vision_encoder_modules_patch_embed_modules_projection_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_patch_embed_modules_projection_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_parameters_pos_embed_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_0_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_0_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_0_modules_mlp_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_0_modules_mlp_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_0_modules_mlp_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_0_modules_mlp_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_1_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_1_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_1_modules_mlp_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_1_modules_mlp_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_1_modules_mlp_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_1_modules_mlp_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_2_modules_mlp_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_2_modules_mlp_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_2_modules_mlp_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_2_modules_mlp_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_3_modules_mlp_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_3_modules_mlp_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_3_modules_mlp_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_3_modules_mlp_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_4_modules_mlp_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_4_modules_mlp_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_4_modules_mlp_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_4_modules_mlp_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_5_modules_mlp_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_5_modules_mlp_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_5_modules_mlp_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_5_modules_mlp_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_6_modules_mlp_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_6_modules_mlp_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_6_modules_mlp_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_6_modules_mlp_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_7_modules_mlp_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_7_modules_mlp_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_7_modules_mlp_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_7_modules_mlp_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_8_modules_mlp_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_8_modules_mlp_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_8_modules_mlp_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_8_modules_mlp_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_9_modules_mlp_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_9_modules_mlp_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_9_modules_mlp_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_9_modules_mlp_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_10_modules_mlp_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_10_modules_mlp_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_10_modules_mlp_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_10_modules_mlp_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_11_modules_mlp_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_11_modules_mlp_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_11_modules_mlp_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_11_modules_mlp_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_12_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_12_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_12_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_12_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_12_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_12_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_12_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_12_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_12_modules_mlp_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_12_modules_mlp_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_12_modules_mlp_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_12_modules_mlp_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_13_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_13_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_13_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_13_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_13_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_13_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_13_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_13_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_13_modules_mlp_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_13_modules_mlp_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_13_modules_mlp_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_13_modules_mlp_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_14_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_14_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_14_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_14_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_14_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_14_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_14_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_14_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_14_modules_mlp_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_14_modules_mlp_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_14_modules_mlp_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_14_modules_mlp_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_15_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_15_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_15_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_15_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_15_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_15_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_15_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_15_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_15_modules_mlp_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_15_modules_mlp_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_15_modules_mlp_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_15_modules_mlp_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_16_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_16_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_16_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_16_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_16_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_16_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_16_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_16_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_16_modules_mlp_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_16_modules_mlp_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_16_modules_mlp_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_16_modules_mlp_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_17_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_17_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_17_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_17_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_17_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_17_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_17_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_17_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_17_modules_mlp_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_17_modules_mlp_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_17_modules_mlp_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_17_modules_mlp_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_18_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_18_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_18_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_18_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_18_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_18_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_18_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_18_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_18_modules_mlp_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_18_modules_mlp_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_18_modules_mlp_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_18_modules_mlp_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_19_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_19_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_19_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_19_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_19_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_19_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_19_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_19_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_19_modules_mlp_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_19_modules_mlp_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_19_modules_mlp_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_19_modules_mlp_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_20_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_20_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_20_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_20_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_20_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_20_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_20_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_20_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_20_modules_mlp_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_20_modules_mlp_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_20_modules_mlp_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_20_modules_mlp_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_21_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_21_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_21_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_21_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_21_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_21_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_21_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_21_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_21_modules_mlp_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_21_modules_mlp_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_21_modules_mlp_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_21_modules_mlp_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_22_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_22_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_22_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_22_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_22_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_22_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_22_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_22_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_22_modules_mlp_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_22_modules_mlp_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_22_modules_mlp_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_22_modules_mlp_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_23_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_23_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_23_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_23_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_23_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_23_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_23_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_23_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_23_modules_mlp_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_23_modules_mlp_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_23_modules_mlp_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_23_modules_mlp_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_24_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_24_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_24_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_24_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_24_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_24_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_24_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_24_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_24_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_24_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_24_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_24_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_24_modules_mlp_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_24_modules_mlp_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_24_modules_mlp_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_24_modules_mlp_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_25_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_25_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_25_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_25_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_25_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_25_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_25_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_25_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_25_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_25_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_25_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_25_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_25_modules_mlp_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_25_modules_mlp_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_25_modules_mlp_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_25_modules_mlp_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_26_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_26_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_26_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_26_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_26_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_26_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_26_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_26_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_26_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_26_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_26_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_26_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_26_modules_mlp_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_26_modules_mlp_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_26_modules_mlp_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_26_modules_mlp_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_27_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_27_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_27_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_27_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_27_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_27_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_27_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_27_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_27_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_27_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_27_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_27_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_27_modules_mlp_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_27_modules_mlp_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_27_modules_mlp_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_27_modules_mlp_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_28_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_28_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_28_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_28_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_28_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_28_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_28_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_28_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_28_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_28_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_28_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_28_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_28_modules_mlp_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_28_modules_mlp_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_28_modules_mlp_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_28_modules_mlp_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_29_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_29_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_29_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_29_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_29_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_29_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_29_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_29_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_29_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_29_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_29_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_29_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_29_modules_mlp_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_29_modules_mlp_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_29_modules_mlp_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_29_modules_mlp_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_30_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_30_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_30_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_30_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_30_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_30_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_30_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_30_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_30_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_30_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_30_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_30_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_30_modules_mlp_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_30_modules_mlp_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_30_modules_mlp_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_30_modules_mlp_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_31_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_31_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_31_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_31_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_31_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_31_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_31_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_31_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_31_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_31_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_31_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_31_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_layers_modules_31_modules_mlp_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_31_modules_mlp_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_31_modules_mlp_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_layers_modules_31_modules_mlp_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_neck_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_neck_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_neck_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_neck_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_neck_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_neck_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_encoder_modules_neck_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_encoder_modules_neck_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_prompt_encoder_modules_no_mask_embed_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_iou_token_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_mask_tokens_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_scaling: torch.Tensor,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_mlp_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_mlp_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_mlp_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_mlp_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm3_eps: torch.Tensor,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_scaling: torch.Tensor,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm4_eps: torch.Tensor,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_scaling: torch.Tensor,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_mlp_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_mlp_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_mlp_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_mlp_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm3_eps: torch.Tensor,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_scaling: torch.Tensor,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm4_eps: torch.Tensor,
        L_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_scaling: torch.Tensor,
        L_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layer_norm_final_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layer_norm_final_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_transformer_modules_layer_norm_final_attn_eps: torch.Tensor,
        L_self_modules_mask_decoder_modules_upscale_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_upscale_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_upscale_layer_norm_eps: torch.Tensor,
        L_self_modules_mask_decoder_modules_upscale_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_upscale_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_upscale_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_upscale_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_proj_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_proj_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_proj_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_proj_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_proj_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_proj_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_proj_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_proj_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_proj_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_proj_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_proj_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_proj_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_proj_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_proj_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_proj_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_proj_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_iou_prediction_head_modules_proj_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_iou_prediction_head_modules_proj_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_iou_prediction_head_modules_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_iou_prediction_head_modules_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_iou_prediction_head_modules_proj_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_decoder_modules_iou_prediction_head_modules_proj_out_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_kwargs_pixel_values_ = L_kwargs_pixel_values_
        l_self_modules_shared_image_embedding_buffers_positional_embedding_ = (
            L_self_modules_shared_image_embedding_buffers_positional_embedding_
        )
        g_import_transformers_dot_models_dot_sam_dot_modeling_sam_np_pi = (
            G_import_transformers_dot_models_dot_sam_dot_modeling_sam_np_pi
        )
        l_self_modules_vision_encoder_modules_patch_embed_modules_projection_parameters_weight_ = L_self_modules_vision_encoder_modules_patch_embed_modules_projection_parameters_weight_
        l_self_modules_vision_encoder_modules_patch_embed_modules_projection_parameters_bias_ = L_self_modules_vision_encoder_modules_patch_embed_modules_projection_parameters_bias_
        l_self_modules_vision_encoder_parameters_pos_embed_ = (
            L_self_modules_vision_encoder_parameters_pos_embed_
        )
        l_self_modules_vision_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_0_modules_layer_norm1_eps = L_self_modules_vision_encoder_modules_layers_modules_0_modules_layer_norm1_eps
        l_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_parameters_rel_pos_h_ = L_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_parameters_rel_pos_h_
        l_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_parameters_rel_pos_w_ = L_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_parameters_rel_pos_w_
        l_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_0_modules_layer_norm2_eps = L_self_modules_vision_encoder_modules_layers_modules_0_modules_layer_norm2_eps
        l_self_modules_vision_encoder_modules_layers_modules_0_modules_mlp_modules_lin1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_0_modules_mlp_modules_lin1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_0_modules_mlp_modules_lin1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_0_modules_mlp_modules_lin1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_0_modules_mlp_modules_lin2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_0_modules_mlp_modules_lin2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_0_modules_mlp_modules_lin2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_0_modules_mlp_modules_lin2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_1_modules_layer_norm1_eps = L_self_modules_vision_encoder_modules_layers_modules_1_modules_layer_norm1_eps
        l_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_parameters_rel_pos_h_ = L_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_parameters_rel_pos_h_
        l_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_parameters_rel_pos_w_ = L_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_parameters_rel_pos_w_
        l_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_modules_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_modules_proj_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_1_modules_layer_norm2_eps = L_self_modules_vision_encoder_modules_layers_modules_1_modules_layer_norm2_eps
        l_self_modules_vision_encoder_modules_layers_modules_1_modules_mlp_modules_lin1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_1_modules_mlp_modules_lin1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_1_modules_mlp_modules_lin1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_1_modules_mlp_modules_lin1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_1_modules_mlp_modules_lin2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_1_modules_mlp_modules_lin2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_1_modules_mlp_modules_lin2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_1_modules_mlp_modules_lin2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm1_eps = L_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm1_eps
        l_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_parameters_rel_pos_h_ = L_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_parameters_rel_pos_h_
        l_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_parameters_rel_pos_w_ = L_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_parameters_rel_pos_w_
        l_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_modules_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_modules_proj_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_modules_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_modules_proj_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm2_eps = L_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm2_eps
        l_self_modules_vision_encoder_modules_layers_modules_2_modules_mlp_modules_lin1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_2_modules_mlp_modules_lin1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_2_modules_mlp_modules_lin1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_2_modules_mlp_modules_lin1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_2_modules_mlp_modules_lin2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_2_modules_mlp_modules_lin2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_2_modules_mlp_modules_lin2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_2_modules_mlp_modules_lin2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm1_eps = L_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm1_eps
        l_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_parameters_rel_pos_h_ = L_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_parameters_rel_pos_h_
        l_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_parameters_rel_pos_w_ = L_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_parameters_rel_pos_w_
        l_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_modules_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_modules_proj_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_modules_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_modules_proj_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm2_eps = L_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm2_eps
        l_self_modules_vision_encoder_modules_layers_modules_3_modules_mlp_modules_lin1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_3_modules_mlp_modules_lin1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_3_modules_mlp_modules_lin1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_3_modules_mlp_modules_lin1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_3_modules_mlp_modules_lin2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_3_modules_mlp_modules_lin2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_3_modules_mlp_modules_lin2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_3_modules_mlp_modules_lin2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm1_eps = L_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm1_eps
        l_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_parameters_rel_pos_h_ = L_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_parameters_rel_pos_h_
        l_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_parameters_rel_pos_w_ = L_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_parameters_rel_pos_w_
        l_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_modules_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_modules_proj_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_modules_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_modules_proj_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm2_eps = L_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm2_eps
        l_self_modules_vision_encoder_modules_layers_modules_4_modules_mlp_modules_lin1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_4_modules_mlp_modules_lin1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_4_modules_mlp_modules_lin1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_4_modules_mlp_modules_lin1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_4_modules_mlp_modules_lin2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_4_modules_mlp_modules_lin2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_4_modules_mlp_modules_lin2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_4_modules_mlp_modules_lin2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm1_eps = L_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm1_eps
        l_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_parameters_rel_pos_h_ = L_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_parameters_rel_pos_h_
        l_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_parameters_rel_pos_w_ = L_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_parameters_rel_pos_w_
        l_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_modules_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_modules_proj_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_modules_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_modules_proj_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm2_eps = L_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm2_eps
        l_self_modules_vision_encoder_modules_layers_modules_5_modules_mlp_modules_lin1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_5_modules_mlp_modules_lin1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_5_modules_mlp_modules_lin1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_5_modules_mlp_modules_lin1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_5_modules_mlp_modules_lin2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_5_modules_mlp_modules_lin2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_5_modules_mlp_modules_lin2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_5_modules_mlp_modules_lin2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm1_eps = L_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm1_eps
        l_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_parameters_rel_pos_h_ = L_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_parameters_rel_pos_h_
        l_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_parameters_rel_pos_w_ = L_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_parameters_rel_pos_w_
        l_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_modules_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_modules_proj_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_modules_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_modules_proj_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm2_eps = L_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm2_eps
        l_self_modules_vision_encoder_modules_layers_modules_6_modules_mlp_modules_lin1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_6_modules_mlp_modules_lin1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_6_modules_mlp_modules_lin1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_6_modules_mlp_modules_lin1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_6_modules_mlp_modules_lin2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_6_modules_mlp_modules_lin2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_6_modules_mlp_modules_lin2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_6_modules_mlp_modules_lin2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm1_eps = L_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm1_eps
        l_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_parameters_rel_pos_h_ = L_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_parameters_rel_pos_h_
        l_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_parameters_rel_pos_w_ = L_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_parameters_rel_pos_w_
        l_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_modules_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_modules_proj_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_modules_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_modules_proj_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm2_eps = L_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm2_eps
        l_self_modules_vision_encoder_modules_layers_modules_7_modules_mlp_modules_lin1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_7_modules_mlp_modules_lin1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_7_modules_mlp_modules_lin1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_7_modules_mlp_modules_lin1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_7_modules_mlp_modules_lin2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_7_modules_mlp_modules_lin2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_7_modules_mlp_modules_lin2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_7_modules_mlp_modules_lin2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm1_eps = L_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm1_eps
        l_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_parameters_rel_pos_h_ = L_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_parameters_rel_pos_h_
        l_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_parameters_rel_pos_w_ = L_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_parameters_rel_pos_w_
        l_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_modules_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_modules_proj_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_modules_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_modules_proj_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm2_eps = L_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm2_eps
        l_self_modules_vision_encoder_modules_layers_modules_8_modules_mlp_modules_lin1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_8_modules_mlp_modules_lin1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_8_modules_mlp_modules_lin1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_8_modules_mlp_modules_lin1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_8_modules_mlp_modules_lin2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_8_modules_mlp_modules_lin2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_8_modules_mlp_modules_lin2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_8_modules_mlp_modules_lin2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm1_eps = L_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm1_eps
        l_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_parameters_rel_pos_h_ = L_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_parameters_rel_pos_h_
        l_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_parameters_rel_pos_w_ = L_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_parameters_rel_pos_w_
        l_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_modules_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_modules_proj_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_modules_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_modules_proj_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm2_eps = L_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm2_eps
        l_self_modules_vision_encoder_modules_layers_modules_9_modules_mlp_modules_lin1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_9_modules_mlp_modules_lin1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_9_modules_mlp_modules_lin1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_9_modules_mlp_modules_lin1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_9_modules_mlp_modules_lin2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_9_modules_mlp_modules_lin2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_9_modules_mlp_modules_lin2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_9_modules_mlp_modules_lin2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm1_eps = L_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm1_eps
        l_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_parameters_rel_pos_h_ = L_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_parameters_rel_pos_h_
        l_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_parameters_rel_pos_w_ = L_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_parameters_rel_pos_w_
        l_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_modules_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_modules_proj_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_modules_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_modules_proj_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm2_eps = L_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm2_eps
        l_self_modules_vision_encoder_modules_layers_modules_10_modules_mlp_modules_lin1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_10_modules_mlp_modules_lin1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_10_modules_mlp_modules_lin1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_10_modules_mlp_modules_lin1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_10_modules_mlp_modules_lin2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_10_modules_mlp_modules_lin2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_10_modules_mlp_modules_lin2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_10_modules_mlp_modules_lin2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm1_eps = L_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm1_eps
        l_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_parameters_rel_pos_h_ = L_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_parameters_rel_pos_h_
        l_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_parameters_rel_pos_w_ = L_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_parameters_rel_pos_w_
        l_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_modules_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_modules_proj_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_modules_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_modules_proj_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm2_eps = L_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm2_eps
        l_self_modules_vision_encoder_modules_layers_modules_11_modules_mlp_modules_lin1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_11_modules_mlp_modules_lin1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_11_modules_mlp_modules_lin1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_11_modules_mlp_modules_lin1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_11_modules_mlp_modules_lin2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_11_modules_mlp_modules_lin2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_11_modules_mlp_modules_lin2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_11_modules_mlp_modules_lin2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_12_modules_layer_norm1_eps = L_self_modules_vision_encoder_modules_layers_modules_12_modules_layer_norm1_eps
        l_self_modules_vision_encoder_modules_layers_modules_12_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_12_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_12_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_12_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_12_modules_attn_parameters_rel_pos_h_ = L_self_modules_vision_encoder_modules_layers_modules_12_modules_attn_parameters_rel_pos_h_
        l_self_modules_vision_encoder_modules_layers_modules_12_modules_attn_parameters_rel_pos_w_ = L_self_modules_vision_encoder_modules_layers_modules_12_modules_attn_parameters_rel_pos_w_
        l_self_modules_vision_encoder_modules_layers_modules_12_modules_attn_modules_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_12_modules_attn_modules_proj_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_12_modules_attn_modules_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_12_modules_attn_modules_proj_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_12_modules_layer_norm2_eps = L_self_modules_vision_encoder_modules_layers_modules_12_modules_layer_norm2_eps
        l_self_modules_vision_encoder_modules_layers_modules_12_modules_mlp_modules_lin1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_12_modules_mlp_modules_lin1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_12_modules_mlp_modules_lin1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_12_modules_mlp_modules_lin1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_12_modules_mlp_modules_lin2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_12_modules_mlp_modules_lin2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_12_modules_mlp_modules_lin2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_12_modules_mlp_modules_lin2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_13_modules_layer_norm1_eps = L_self_modules_vision_encoder_modules_layers_modules_13_modules_layer_norm1_eps
        l_self_modules_vision_encoder_modules_layers_modules_13_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_13_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_13_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_13_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_13_modules_attn_parameters_rel_pos_h_ = L_self_modules_vision_encoder_modules_layers_modules_13_modules_attn_parameters_rel_pos_h_
        l_self_modules_vision_encoder_modules_layers_modules_13_modules_attn_parameters_rel_pos_w_ = L_self_modules_vision_encoder_modules_layers_modules_13_modules_attn_parameters_rel_pos_w_
        l_self_modules_vision_encoder_modules_layers_modules_13_modules_attn_modules_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_13_modules_attn_modules_proj_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_13_modules_attn_modules_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_13_modules_attn_modules_proj_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_13_modules_layer_norm2_eps = L_self_modules_vision_encoder_modules_layers_modules_13_modules_layer_norm2_eps
        l_self_modules_vision_encoder_modules_layers_modules_13_modules_mlp_modules_lin1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_13_modules_mlp_modules_lin1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_13_modules_mlp_modules_lin1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_13_modules_mlp_modules_lin1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_13_modules_mlp_modules_lin2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_13_modules_mlp_modules_lin2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_13_modules_mlp_modules_lin2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_13_modules_mlp_modules_lin2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_14_modules_layer_norm1_eps = L_self_modules_vision_encoder_modules_layers_modules_14_modules_layer_norm1_eps
        l_self_modules_vision_encoder_modules_layers_modules_14_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_14_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_14_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_14_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_14_modules_attn_parameters_rel_pos_h_ = L_self_modules_vision_encoder_modules_layers_modules_14_modules_attn_parameters_rel_pos_h_
        l_self_modules_vision_encoder_modules_layers_modules_14_modules_attn_parameters_rel_pos_w_ = L_self_modules_vision_encoder_modules_layers_modules_14_modules_attn_parameters_rel_pos_w_
        l_self_modules_vision_encoder_modules_layers_modules_14_modules_attn_modules_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_14_modules_attn_modules_proj_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_14_modules_attn_modules_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_14_modules_attn_modules_proj_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_14_modules_layer_norm2_eps = L_self_modules_vision_encoder_modules_layers_modules_14_modules_layer_norm2_eps
        l_self_modules_vision_encoder_modules_layers_modules_14_modules_mlp_modules_lin1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_14_modules_mlp_modules_lin1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_14_modules_mlp_modules_lin1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_14_modules_mlp_modules_lin1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_14_modules_mlp_modules_lin2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_14_modules_mlp_modules_lin2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_14_modules_mlp_modules_lin2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_14_modules_mlp_modules_lin2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_15_modules_layer_norm1_eps = L_self_modules_vision_encoder_modules_layers_modules_15_modules_layer_norm1_eps
        l_self_modules_vision_encoder_modules_layers_modules_15_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_15_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_15_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_15_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_15_modules_attn_parameters_rel_pos_h_ = L_self_modules_vision_encoder_modules_layers_modules_15_modules_attn_parameters_rel_pos_h_
        l_self_modules_vision_encoder_modules_layers_modules_15_modules_attn_parameters_rel_pos_w_ = L_self_modules_vision_encoder_modules_layers_modules_15_modules_attn_parameters_rel_pos_w_
        l_self_modules_vision_encoder_modules_layers_modules_15_modules_attn_modules_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_15_modules_attn_modules_proj_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_15_modules_attn_modules_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_15_modules_attn_modules_proj_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_15_modules_layer_norm2_eps = L_self_modules_vision_encoder_modules_layers_modules_15_modules_layer_norm2_eps
        l_self_modules_vision_encoder_modules_layers_modules_15_modules_mlp_modules_lin1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_15_modules_mlp_modules_lin1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_15_modules_mlp_modules_lin1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_15_modules_mlp_modules_lin1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_15_modules_mlp_modules_lin2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_15_modules_mlp_modules_lin2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_15_modules_mlp_modules_lin2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_15_modules_mlp_modules_lin2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_16_modules_layer_norm1_eps = L_self_modules_vision_encoder_modules_layers_modules_16_modules_layer_norm1_eps
        l_self_modules_vision_encoder_modules_layers_modules_16_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_16_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_16_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_16_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_16_modules_attn_parameters_rel_pos_h_ = L_self_modules_vision_encoder_modules_layers_modules_16_modules_attn_parameters_rel_pos_h_
        l_self_modules_vision_encoder_modules_layers_modules_16_modules_attn_parameters_rel_pos_w_ = L_self_modules_vision_encoder_modules_layers_modules_16_modules_attn_parameters_rel_pos_w_
        l_self_modules_vision_encoder_modules_layers_modules_16_modules_attn_modules_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_16_modules_attn_modules_proj_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_16_modules_attn_modules_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_16_modules_attn_modules_proj_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_16_modules_layer_norm2_eps = L_self_modules_vision_encoder_modules_layers_modules_16_modules_layer_norm2_eps
        l_self_modules_vision_encoder_modules_layers_modules_16_modules_mlp_modules_lin1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_16_modules_mlp_modules_lin1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_16_modules_mlp_modules_lin1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_16_modules_mlp_modules_lin1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_16_modules_mlp_modules_lin2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_16_modules_mlp_modules_lin2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_16_modules_mlp_modules_lin2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_16_modules_mlp_modules_lin2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_17_modules_layer_norm1_eps = L_self_modules_vision_encoder_modules_layers_modules_17_modules_layer_norm1_eps
        l_self_modules_vision_encoder_modules_layers_modules_17_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_17_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_17_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_17_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_17_modules_attn_parameters_rel_pos_h_ = L_self_modules_vision_encoder_modules_layers_modules_17_modules_attn_parameters_rel_pos_h_
        l_self_modules_vision_encoder_modules_layers_modules_17_modules_attn_parameters_rel_pos_w_ = L_self_modules_vision_encoder_modules_layers_modules_17_modules_attn_parameters_rel_pos_w_
        l_self_modules_vision_encoder_modules_layers_modules_17_modules_attn_modules_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_17_modules_attn_modules_proj_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_17_modules_attn_modules_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_17_modules_attn_modules_proj_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_17_modules_layer_norm2_eps = L_self_modules_vision_encoder_modules_layers_modules_17_modules_layer_norm2_eps
        l_self_modules_vision_encoder_modules_layers_modules_17_modules_mlp_modules_lin1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_17_modules_mlp_modules_lin1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_17_modules_mlp_modules_lin1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_17_modules_mlp_modules_lin1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_17_modules_mlp_modules_lin2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_17_modules_mlp_modules_lin2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_17_modules_mlp_modules_lin2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_17_modules_mlp_modules_lin2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_18_modules_layer_norm1_eps = L_self_modules_vision_encoder_modules_layers_modules_18_modules_layer_norm1_eps
        l_self_modules_vision_encoder_modules_layers_modules_18_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_18_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_18_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_18_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_18_modules_attn_parameters_rel_pos_h_ = L_self_modules_vision_encoder_modules_layers_modules_18_modules_attn_parameters_rel_pos_h_
        l_self_modules_vision_encoder_modules_layers_modules_18_modules_attn_parameters_rel_pos_w_ = L_self_modules_vision_encoder_modules_layers_modules_18_modules_attn_parameters_rel_pos_w_
        l_self_modules_vision_encoder_modules_layers_modules_18_modules_attn_modules_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_18_modules_attn_modules_proj_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_18_modules_attn_modules_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_18_modules_attn_modules_proj_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_18_modules_layer_norm2_eps = L_self_modules_vision_encoder_modules_layers_modules_18_modules_layer_norm2_eps
        l_self_modules_vision_encoder_modules_layers_modules_18_modules_mlp_modules_lin1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_18_modules_mlp_modules_lin1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_18_modules_mlp_modules_lin1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_18_modules_mlp_modules_lin1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_18_modules_mlp_modules_lin2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_18_modules_mlp_modules_lin2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_18_modules_mlp_modules_lin2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_18_modules_mlp_modules_lin2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_19_modules_layer_norm1_eps = L_self_modules_vision_encoder_modules_layers_modules_19_modules_layer_norm1_eps
        l_self_modules_vision_encoder_modules_layers_modules_19_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_19_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_19_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_19_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_19_modules_attn_parameters_rel_pos_h_ = L_self_modules_vision_encoder_modules_layers_modules_19_modules_attn_parameters_rel_pos_h_
        l_self_modules_vision_encoder_modules_layers_modules_19_modules_attn_parameters_rel_pos_w_ = L_self_modules_vision_encoder_modules_layers_modules_19_modules_attn_parameters_rel_pos_w_
        l_self_modules_vision_encoder_modules_layers_modules_19_modules_attn_modules_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_19_modules_attn_modules_proj_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_19_modules_attn_modules_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_19_modules_attn_modules_proj_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_19_modules_layer_norm2_eps = L_self_modules_vision_encoder_modules_layers_modules_19_modules_layer_norm2_eps
        l_self_modules_vision_encoder_modules_layers_modules_19_modules_mlp_modules_lin1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_19_modules_mlp_modules_lin1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_19_modules_mlp_modules_lin1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_19_modules_mlp_modules_lin1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_19_modules_mlp_modules_lin2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_19_modules_mlp_modules_lin2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_19_modules_mlp_modules_lin2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_19_modules_mlp_modules_lin2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_20_modules_layer_norm1_eps = L_self_modules_vision_encoder_modules_layers_modules_20_modules_layer_norm1_eps
        l_self_modules_vision_encoder_modules_layers_modules_20_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_20_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_20_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_20_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_20_modules_attn_parameters_rel_pos_h_ = L_self_modules_vision_encoder_modules_layers_modules_20_modules_attn_parameters_rel_pos_h_
        l_self_modules_vision_encoder_modules_layers_modules_20_modules_attn_parameters_rel_pos_w_ = L_self_modules_vision_encoder_modules_layers_modules_20_modules_attn_parameters_rel_pos_w_
        l_self_modules_vision_encoder_modules_layers_modules_20_modules_attn_modules_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_20_modules_attn_modules_proj_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_20_modules_attn_modules_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_20_modules_attn_modules_proj_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_20_modules_layer_norm2_eps = L_self_modules_vision_encoder_modules_layers_modules_20_modules_layer_norm2_eps
        l_self_modules_vision_encoder_modules_layers_modules_20_modules_mlp_modules_lin1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_20_modules_mlp_modules_lin1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_20_modules_mlp_modules_lin1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_20_modules_mlp_modules_lin1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_20_modules_mlp_modules_lin2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_20_modules_mlp_modules_lin2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_20_modules_mlp_modules_lin2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_20_modules_mlp_modules_lin2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_21_modules_layer_norm1_eps = L_self_modules_vision_encoder_modules_layers_modules_21_modules_layer_norm1_eps
        l_self_modules_vision_encoder_modules_layers_modules_21_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_21_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_21_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_21_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_21_modules_attn_parameters_rel_pos_h_ = L_self_modules_vision_encoder_modules_layers_modules_21_modules_attn_parameters_rel_pos_h_
        l_self_modules_vision_encoder_modules_layers_modules_21_modules_attn_parameters_rel_pos_w_ = L_self_modules_vision_encoder_modules_layers_modules_21_modules_attn_parameters_rel_pos_w_
        l_self_modules_vision_encoder_modules_layers_modules_21_modules_attn_modules_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_21_modules_attn_modules_proj_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_21_modules_attn_modules_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_21_modules_attn_modules_proj_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_21_modules_layer_norm2_eps = L_self_modules_vision_encoder_modules_layers_modules_21_modules_layer_norm2_eps
        l_self_modules_vision_encoder_modules_layers_modules_21_modules_mlp_modules_lin1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_21_modules_mlp_modules_lin1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_21_modules_mlp_modules_lin1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_21_modules_mlp_modules_lin1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_21_modules_mlp_modules_lin2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_21_modules_mlp_modules_lin2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_21_modules_mlp_modules_lin2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_21_modules_mlp_modules_lin2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_22_modules_layer_norm1_eps = L_self_modules_vision_encoder_modules_layers_modules_22_modules_layer_norm1_eps
        l_self_modules_vision_encoder_modules_layers_modules_22_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_22_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_22_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_22_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_22_modules_attn_parameters_rel_pos_h_ = L_self_modules_vision_encoder_modules_layers_modules_22_modules_attn_parameters_rel_pos_h_
        l_self_modules_vision_encoder_modules_layers_modules_22_modules_attn_parameters_rel_pos_w_ = L_self_modules_vision_encoder_modules_layers_modules_22_modules_attn_parameters_rel_pos_w_
        l_self_modules_vision_encoder_modules_layers_modules_22_modules_attn_modules_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_22_modules_attn_modules_proj_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_22_modules_attn_modules_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_22_modules_attn_modules_proj_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_22_modules_layer_norm2_eps = L_self_modules_vision_encoder_modules_layers_modules_22_modules_layer_norm2_eps
        l_self_modules_vision_encoder_modules_layers_modules_22_modules_mlp_modules_lin1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_22_modules_mlp_modules_lin1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_22_modules_mlp_modules_lin1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_22_modules_mlp_modules_lin1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_22_modules_mlp_modules_lin2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_22_modules_mlp_modules_lin2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_22_modules_mlp_modules_lin2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_22_modules_mlp_modules_lin2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_23_modules_layer_norm1_eps = L_self_modules_vision_encoder_modules_layers_modules_23_modules_layer_norm1_eps
        l_self_modules_vision_encoder_modules_layers_modules_23_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_23_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_23_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_23_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_23_modules_attn_parameters_rel_pos_h_ = L_self_modules_vision_encoder_modules_layers_modules_23_modules_attn_parameters_rel_pos_h_
        l_self_modules_vision_encoder_modules_layers_modules_23_modules_attn_parameters_rel_pos_w_ = L_self_modules_vision_encoder_modules_layers_modules_23_modules_attn_parameters_rel_pos_w_
        l_self_modules_vision_encoder_modules_layers_modules_23_modules_attn_modules_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_23_modules_attn_modules_proj_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_23_modules_attn_modules_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_23_modules_attn_modules_proj_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_23_modules_layer_norm2_eps = L_self_modules_vision_encoder_modules_layers_modules_23_modules_layer_norm2_eps
        l_self_modules_vision_encoder_modules_layers_modules_23_modules_mlp_modules_lin1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_23_modules_mlp_modules_lin1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_23_modules_mlp_modules_lin1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_23_modules_mlp_modules_lin1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_23_modules_mlp_modules_lin2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_23_modules_mlp_modules_lin2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_23_modules_mlp_modules_lin2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_23_modules_mlp_modules_lin2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_24_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_24_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_24_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_24_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_24_modules_layer_norm1_eps = L_self_modules_vision_encoder_modules_layers_modules_24_modules_layer_norm1_eps
        l_self_modules_vision_encoder_modules_layers_modules_24_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_24_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_24_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_24_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_24_modules_attn_parameters_rel_pos_h_ = L_self_modules_vision_encoder_modules_layers_modules_24_modules_attn_parameters_rel_pos_h_
        l_self_modules_vision_encoder_modules_layers_modules_24_modules_attn_parameters_rel_pos_w_ = L_self_modules_vision_encoder_modules_layers_modules_24_modules_attn_parameters_rel_pos_w_
        l_self_modules_vision_encoder_modules_layers_modules_24_modules_attn_modules_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_24_modules_attn_modules_proj_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_24_modules_attn_modules_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_24_modules_attn_modules_proj_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_24_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_24_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_24_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_24_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_24_modules_layer_norm2_eps = L_self_modules_vision_encoder_modules_layers_modules_24_modules_layer_norm2_eps
        l_self_modules_vision_encoder_modules_layers_modules_24_modules_mlp_modules_lin1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_24_modules_mlp_modules_lin1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_24_modules_mlp_modules_lin1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_24_modules_mlp_modules_lin1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_24_modules_mlp_modules_lin2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_24_modules_mlp_modules_lin2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_24_modules_mlp_modules_lin2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_24_modules_mlp_modules_lin2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_25_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_25_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_25_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_25_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_25_modules_layer_norm1_eps = L_self_modules_vision_encoder_modules_layers_modules_25_modules_layer_norm1_eps
        l_self_modules_vision_encoder_modules_layers_modules_25_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_25_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_25_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_25_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_25_modules_attn_parameters_rel_pos_h_ = L_self_modules_vision_encoder_modules_layers_modules_25_modules_attn_parameters_rel_pos_h_
        l_self_modules_vision_encoder_modules_layers_modules_25_modules_attn_parameters_rel_pos_w_ = L_self_modules_vision_encoder_modules_layers_modules_25_modules_attn_parameters_rel_pos_w_
        l_self_modules_vision_encoder_modules_layers_modules_25_modules_attn_modules_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_25_modules_attn_modules_proj_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_25_modules_attn_modules_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_25_modules_attn_modules_proj_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_25_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_25_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_25_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_25_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_25_modules_layer_norm2_eps = L_self_modules_vision_encoder_modules_layers_modules_25_modules_layer_norm2_eps
        l_self_modules_vision_encoder_modules_layers_modules_25_modules_mlp_modules_lin1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_25_modules_mlp_modules_lin1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_25_modules_mlp_modules_lin1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_25_modules_mlp_modules_lin1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_25_modules_mlp_modules_lin2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_25_modules_mlp_modules_lin2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_25_modules_mlp_modules_lin2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_25_modules_mlp_modules_lin2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_26_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_26_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_26_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_26_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_26_modules_layer_norm1_eps = L_self_modules_vision_encoder_modules_layers_modules_26_modules_layer_norm1_eps
        l_self_modules_vision_encoder_modules_layers_modules_26_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_26_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_26_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_26_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_26_modules_attn_parameters_rel_pos_h_ = L_self_modules_vision_encoder_modules_layers_modules_26_modules_attn_parameters_rel_pos_h_
        l_self_modules_vision_encoder_modules_layers_modules_26_modules_attn_parameters_rel_pos_w_ = L_self_modules_vision_encoder_modules_layers_modules_26_modules_attn_parameters_rel_pos_w_
        l_self_modules_vision_encoder_modules_layers_modules_26_modules_attn_modules_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_26_modules_attn_modules_proj_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_26_modules_attn_modules_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_26_modules_attn_modules_proj_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_26_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_26_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_26_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_26_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_26_modules_layer_norm2_eps = L_self_modules_vision_encoder_modules_layers_modules_26_modules_layer_norm2_eps
        l_self_modules_vision_encoder_modules_layers_modules_26_modules_mlp_modules_lin1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_26_modules_mlp_modules_lin1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_26_modules_mlp_modules_lin1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_26_modules_mlp_modules_lin1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_26_modules_mlp_modules_lin2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_26_modules_mlp_modules_lin2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_26_modules_mlp_modules_lin2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_26_modules_mlp_modules_lin2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_27_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_27_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_27_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_27_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_27_modules_layer_norm1_eps = L_self_modules_vision_encoder_modules_layers_modules_27_modules_layer_norm1_eps
        l_self_modules_vision_encoder_modules_layers_modules_27_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_27_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_27_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_27_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_27_modules_attn_parameters_rel_pos_h_ = L_self_modules_vision_encoder_modules_layers_modules_27_modules_attn_parameters_rel_pos_h_
        l_self_modules_vision_encoder_modules_layers_modules_27_modules_attn_parameters_rel_pos_w_ = L_self_modules_vision_encoder_modules_layers_modules_27_modules_attn_parameters_rel_pos_w_
        l_self_modules_vision_encoder_modules_layers_modules_27_modules_attn_modules_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_27_modules_attn_modules_proj_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_27_modules_attn_modules_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_27_modules_attn_modules_proj_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_27_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_27_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_27_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_27_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_27_modules_layer_norm2_eps = L_self_modules_vision_encoder_modules_layers_modules_27_modules_layer_norm2_eps
        l_self_modules_vision_encoder_modules_layers_modules_27_modules_mlp_modules_lin1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_27_modules_mlp_modules_lin1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_27_modules_mlp_modules_lin1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_27_modules_mlp_modules_lin1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_27_modules_mlp_modules_lin2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_27_modules_mlp_modules_lin2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_27_modules_mlp_modules_lin2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_27_modules_mlp_modules_lin2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_28_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_28_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_28_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_28_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_28_modules_layer_norm1_eps = L_self_modules_vision_encoder_modules_layers_modules_28_modules_layer_norm1_eps
        l_self_modules_vision_encoder_modules_layers_modules_28_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_28_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_28_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_28_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_28_modules_attn_parameters_rel_pos_h_ = L_self_modules_vision_encoder_modules_layers_modules_28_modules_attn_parameters_rel_pos_h_
        l_self_modules_vision_encoder_modules_layers_modules_28_modules_attn_parameters_rel_pos_w_ = L_self_modules_vision_encoder_modules_layers_modules_28_modules_attn_parameters_rel_pos_w_
        l_self_modules_vision_encoder_modules_layers_modules_28_modules_attn_modules_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_28_modules_attn_modules_proj_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_28_modules_attn_modules_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_28_modules_attn_modules_proj_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_28_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_28_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_28_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_28_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_28_modules_layer_norm2_eps = L_self_modules_vision_encoder_modules_layers_modules_28_modules_layer_norm2_eps
        l_self_modules_vision_encoder_modules_layers_modules_28_modules_mlp_modules_lin1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_28_modules_mlp_modules_lin1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_28_modules_mlp_modules_lin1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_28_modules_mlp_modules_lin1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_28_modules_mlp_modules_lin2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_28_modules_mlp_modules_lin2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_28_modules_mlp_modules_lin2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_28_modules_mlp_modules_lin2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_29_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_29_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_29_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_29_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_29_modules_layer_norm1_eps = L_self_modules_vision_encoder_modules_layers_modules_29_modules_layer_norm1_eps
        l_self_modules_vision_encoder_modules_layers_modules_29_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_29_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_29_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_29_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_29_modules_attn_parameters_rel_pos_h_ = L_self_modules_vision_encoder_modules_layers_modules_29_modules_attn_parameters_rel_pos_h_
        l_self_modules_vision_encoder_modules_layers_modules_29_modules_attn_parameters_rel_pos_w_ = L_self_modules_vision_encoder_modules_layers_modules_29_modules_attn_parameters_rel_pos_w_
        l_self_modules_vision_encoder_modules_layers_modules_29_modules_attn_modules_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_29_modules_attn_modules_proj_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_29_modules_attn_modules_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_29_modules_attn_modules_proj_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_29_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_29_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_29_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_29_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_29_modules_layer_norm2_eps = L_self_modules_vision_encoder_modules_layers_modules_29_modules_layer_norm2_eps
        l_self_modules_vision_encoder_modules_layers_modules_29_modules_mlp_modules_lin1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_29_modules_mlp_modules_lin1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_29_modules_mlp_modules_lin1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_29_modules_mlp_modules_lin1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_29_modules_mlp_modules_lin2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_29_modules_mlp_modules_lin2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_29_modules_mlp_modules_lin2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_29_modules_mlp_modules_lin2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_30_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_30_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_30_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_30_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_30_modules_layer_norm1_eps = L_self_modules_vision_encoder_modules_layers_modules_30_modules_layer_norm1_eps
        l_self_modules_vision_encoder_modules_layers_modules_30_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_30_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_30_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_30_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_30_modules_attn_parameters_rel_pos_h_ = L_self_modules_vision_encoder_modules_layers_modules_30_modules_attn_parameters_rel_pos_h_
        l_self_modules_vision_encoder_modules_layers_modules_30_modules_attn_parameters_rel_pos_w_ = L_self_modules_vision_encoder_modules_layers_modules_30_modules_attn_parameters_rel_pos_w_
        l_self_modules_vision_encoder_modules_layers_modules_30_modules_attn_modules_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_30_modules_attn_modules_proj_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_30_modules_attn_modules_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_30_modules_attn_modules_proj_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_30_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_30_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_30_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_30_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_30_modules_layer_norm2_eps = L_self_modules_vision_encoder_modules_layers_modules_30_modules_layer_norm2_eps
        l_self_modules_vision_encoder_modules_layers_modules_30_modules_mlp_modules_lin1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_30_modules_mlp_modules_lin1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_30_modules_mlp_modules_lin1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_30_modules_mlp_modules_lin1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_30_modules_mlp_modules_lin2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_30_modules_mlp_modules_lin2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_30_modules_mlp_modules_lin2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_30_modules_mlp_modules_lin2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_31_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_31_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_31_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_31_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_31_modules_layer_norm1_eps = L_self_modules_vision_encoder_modules_layers_modules_31_modules_layer_norm1_eps
        l_self_modules_vision_encoder_modules_layers_modules_31_modules_attn_modules_qkv_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_31_modules_attn_modules_qkv_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_31_modules_attn_modules_qkv_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_31_modules_attn_modules_qkv_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_31_modules_attn_parameters_rel_pos_h_ = L_self_modules_vision_encoder_modules_layers_modules_31_modules_attn_parameters_rel_pos_h_
        l_self_modules_vision_encoder_modules_layers_modules_31_modules_attn_parameters_rel_pos_w_ = L_self_modules_vision_encoder_modules_layers_modules_31_modules_attn_parameters_rel_pos_w_
        l_self_modules_vision_encoder_modules_layers_modules_31_modules_attn_modules_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_31_modules_attn_modules_proj_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_31_modules_attn_modules_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_31_modules_attn_modules_proj_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_31_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_31_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_31_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_31_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_31_modules_layer_norm2_eps = L_self_modules_vision_encoder_modules_layers_modules_31_modules_layer_norm2_eps
        l_self_modules_vision_encoder_modules_layers_modules_31_modules_mlp_modules_lin1_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_31_modules_mlp_modules_lin1_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_31_modules_mlp_modules_lin1_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_31_modules_mlp_modules_lin1_parameters_bias_
        l_self_modules_vision_encoder_modules_layers_modules_31_modules_mlp_modules_lin2_parameters_weight_ = L_self_modules_vision_encoder_modules_layers_modules_31_modules_mlp_modules_lin2_parameters_weight_
        l_self_modules_vision_encoder_modules_layers_modules_31_modules_mlp_modules_lin2_parameters_bias_ = L_self_modules_vision_encoder_modules_layers_modules_31_modules_mlp_modules_lin2_parameters_bias_
        l_self_modules_vision_encoder_modules_neck_modules_conv1_parameters_weight_ = (
            L_self_modules_vision_encoder_modules_neck_modules_conv1_parameters_weight_
        )
        l_self_modules_vision_encoder_modules_neck_modules_layer_norm1_eps = (
            L_self_modules_vision_encoder_modules_neck_modules_layer_norm1_eps
        )
        l_self_modules_vision_encoder_modules_neck_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_neck_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_encoder_modules_neck_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_neck_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_encoder_modules_neck_modules_conv2_parameters_weight_ = (
            L_self_modules_vision_encoder_modules_neck_modules_conv2_parameters_weight_
        )
        l_self_modules_vision_encoder_modules_neck_modules_layer_norm2_eps = (
            L_self_modules_vision_encoder_modules_neck_modules_layer_norm2_eps
        )
        l_self_modules_vision_encoder_modules_neck_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_neck_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_encoder_modules_neck_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_neck_modules_layer_norm2_parameters_bias_
        l_self_modules_prompt_encoder_modules_no_mask_embed_parameters_weight_ = (
            L_self_modules_prompt_encoder_modules_no_mask_embed_parameters_weight_
        )
        l_self_modules_mask_decoder_modules_iou_token_parameters_weight_ = (
            L_self_modules_mask_decoder_modules_iou_token_parameters_weight_
        )
        l_self_modules_mask_decoder_modules_mask_tokens_parameters_weight_ = (
            L_self_modules_mask_decoder_modules_mask_tokens_parameters_weight_
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_scaling = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_scaling
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm1_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm1_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm1_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm1_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm1_eps = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm1_eps
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_q_proj_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_q_proj_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_q_proj_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_q_proj_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_k_proj_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_k_proj_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_k_proj_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_k_proj_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_v_proj_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_v_proj_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_v_proj_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_v_proj_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_scaling = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_scaling
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_out_proj_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_out_proj_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_out_proj_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_out_proj_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm2_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm2_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm2_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm2_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm2_eps = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm2_eps
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_mlp_modules_lin1_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_mlp_modules_lin1_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_mlp_modules_lin1_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_mlp_modules_lin1_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_mlp_modules_lin2_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_mlp_modules_lin2_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_mlp_modules_lin2_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_mlp_modules_lin2_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm3_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm3_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm3_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm3_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm3_eps = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm3_eps
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_q_proj_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_q_proj_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_q_proj_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_q_proj_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_k_proj_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_k_proj_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_k_proj_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_k_proj_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_v_proj_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_v_proj_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_v_proj_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_v_proj_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_scaling = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_scaling
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_out_proj_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_out_proj_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_out_proj_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_out_proj_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm4_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm4_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm4_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm4_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm4_eps = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm4_eps
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_scaling = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_scaling
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm1_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm1_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm1_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm1_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm1_eps = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm1_eps
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_q_proj_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_q_proj_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_q_proj_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_q_proj_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_k_proj_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_k_proj_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_k_proj_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_k_proj_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_v_proj_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_v_proj_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_v_proj_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_v_proj_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_scaling = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_scaling
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_out_proj_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_out_proj_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_out_proj_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_out_proj_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm2_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm2_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm2_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm2_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm2_eps = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm2_eps
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_mlp_modules_lin1_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_mlp_modules_lin1_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_mlp_modules_lin1_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_mlp_modules_lin1_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_mlp_modules_lin2_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_mlp_modules_lin2_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_mlp_modules_lin2_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_mlp_modules_lin2_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm3_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm3_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm3_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm3_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm3_eps = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm3_eps
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_q_proj_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_q_proj_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_q_proj_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_q_proj_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_k_proj_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_k_proj_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_k_proj_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_k_proj_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_v_proj_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_v_proj_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_v_proj_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_v_proj_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_scaling = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_scaling
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_out_proj_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_out_proj_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_out_proj_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_out_proj_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm4_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm4_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm4_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm4_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm4_eps = L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm4_eps
        l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_q_proj_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_q_proj_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_q_proj_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_q_proj_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_k_proj_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_k_proj_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_k_proj_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_k_proj_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_v_proj_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_v_proj_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_v_proj_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_v_proj_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_scaling = L_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_scaling
        l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_out_proj_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_out_proj_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_out_proj_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_out_proj_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_layer_norm_final_attn_parameters_weight_ = L_self_modules_mask_decoder_modules_transformer_modules_layer_norm_final_attn_parameters_weight_
        l_self_modules_mask_decoder_modules_transformer_modules_layer_norm_final_attn_parameters_bias_ = L_self_modules_mask_decoder_modules_transformer_modules_layer_norm_final_attn_parameters_bias_
        l_self_modules_mask_decoder_modules_transformer_modules_layer_norm_final_attn_eps = L_self_modules_mask_decoder_modules_transformer_modules_layer_norm_final_attn_eps
        l_self_modules_mask_decoder_modules_upscale_conv1_parameters_weight_ = (
            L_self_modules_mask_decoder_modules_upscale_conv1_parameters_weight_
        )
        l_self_modules_mask_decoder_modules_upscale_conv1_parameters_bias_ = (
            L_self_modules_mask_decoder_modules_upscale_conv1_parameters_bias_
        )
        l_self_modules_mask_decoder_modules_upscale_layer_norm_eps = (
            L_self_modules_mask_decoder_modules_upscale_layer_norm_eps
        )
        l_self_modules_mask_decoder_modules_upscale_layer_norm_parameters_weight_ = (
            L_self_modules_mask_decoder_modules_upscale_layer_norm_parameters_weight_
        )
        l_self_modules_mask_decoder_modules_upscale_layer_norm_parameters_bias_ = (
            L_self_modules_mask_decoder_modules_upscale_layer_norm_parameters_bias_
        )
        l_self_modules_mask_decoder_modules_upscale_conv2_parameters_weight_ = (
            L_self_modules_mask_decoder_modules_upscale_conv2_parameters_weight_
        )
        l_self_modules_mask_decoder_modules_upscale_conv2_parameters_bias_ = (
            L_self_modules_mask_decoder_modules_upscale_conv2_parameters_bias_
        )
        l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_proj_in_parameters_weight_ = L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_proj_in_parameters_weight_
        l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_proj_in_parameters_bias_ = L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_proj_in_parameters_bias_
        l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_layers_modules_0_parameters_weight_ = L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_layers_modules_0_parameters_weight_
        l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_layers_modules_0_parameters_bias_ = L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_layers_modules_0_parameters_bias_
        l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_proj_out_parameters_weight_ = L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_proj_out_parameters_weight_
        l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_proj_out_parameters_bias_ = L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_proj_out_parameters_bias_
        l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_proj_in_parameters_weight_ = L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_proj_in_parameters_weight_
        l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_proj_in_parameters_bias_ = L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_proj_in_parameters_bias_
        l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_layers_modules_0_parameters_weight_ = L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_layers_modules_0_parameters_weight_
        l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_layers_modules_0_parameters_bias_ = L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_layers_modules_0_parameters_bias_
        l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_proj_out_parameters_weight_ = L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_proj_out_parameters_weight_
        l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_proj_out_parameters_bias_ = L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_proj_out_parameters_bias_
        l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_proj_in_parameters_weight_ = L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_proj_in_parameters_weight_
        l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_proj_in_parameters_bias_ = L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_proj_in_parameters_bias_
        l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_layers_modules_0_parameters_weight_ = L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_layers_modules_0_parameters_weight_
        l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_layers_modules_0_parameters_bias_ = L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_layers_modules_0_parameters_bias_
        l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_proj_out_parameters_weight_ = L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_proj_out_parameters_weight_
        l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_proj_out_parameters_bias_ = L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_proj_out_parameters_bias_
        l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_proj_in_parameters_weight_ = L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_proj_in_parameters_weight_
        l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_proj_in_parameters_bias_ = L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_proj_in_parameters_bias_
        l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_layers_modules_0_parameters_weight_ = L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_layers_modules_0_parameters_weight_
        l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_layers_modules_0_parameters_bias_ = L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_layers_modules_0_parameters_bias_
        l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_proj_out_parameters_weight_ = L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_proj_out_parameters_weight_
        l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_proj_out_parameters_bias_ = L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_proj_out_parameters_bias_
        l_self_modules_mask_decoder_modules_iou_prediction_head_modules_proj_in_parameters_weight_ = L_self_modules_mask_decoder_modules_iou_prediction_head_modules_proj_in_parameters_weight_
        l_self_modules_mask_decoder_modules_iou_prediction_head_modules_proj_in_parameters_bias_ = L_self_modules_mask_decoder_modules_iou_prediction_head_modules_proj_in_parameters_bias_
        l_self_modules_mask_decoder_modules_iou_prediction_head_modules_layers_modules_0_parameters_weight_ = L_self_modules_mask_decoder_modules_iou_prediction_head_modules_layers_modules_0_parameters_weight_
        l_self_modules_mask_decoder_modules_iou_prediction_head_modules_layers_modules_0_parameters_bias_ = L_self_modules_mask_decoder_modules_iou_prediction_head_modules_layers_modules_0_parameters_bias_
        l_self_modules_mask_decoder_modules_iou_prediction_head_modules_proj_out_parameters_weight_ = L_self_modules_mask_decoder_modules_iou_prediction_head_modules_proj_out_parameters_weight_
        l_self_modules_mask_decoder_modules_iou_prediction_head_modules_proj_out_parameters_bias_ = L_self_modules_mask_decoder_modules_iou_prediction_head_modules_proj_out_parameters_bias_
        grid = torch.ones(
            (64, 64), device=device(type="cuda", index=0), dtype=torch.float32
        )
        cumsum = grid.cumsum(dim=0)
        y_embed = cumsum - 0.5
        cumsum = None
        cumsum_1 = grid.cumsum(dim=1)
        grid = None
        x_embed = cumsum_1 - 0.5
        cumsum_1 = None
        y_embed_1 = y_embed / 64
        y_embed = None
        x_embed_1 = x_embed / 64
        x_embed = None
        stack = torch.stack([x_embed_1, y_embed_1], dim=-1)
        x_embed_1 = y_embed_1 = None
        coordinates = stack.clone()
        stack = None
        mul = 2 * coordinates
        coordinates = None
        coordinates_1 = mul - 1
        mul = None
        coordinates_2 = coordinates_1.to(torch.float32)
        coordinates_1 = None
        coordinates_3 = (
            coordinates_2
            @ l_self_modules_shared_image_embedding_buffers_positional_embedding_
        )
        coordinates_2 = (
            l_self_modules_shared_image_embedding_buffers_positional_embedding_
        ) = None
        item = g_import_transformers_dot_models_dot_sam_dot_modeling_sam_np_pi.item()
        g_import_transformers_dot_models_dot_sam_dot_modeling_sam_np_pi = None
        mul_1 = 2 * item
        item = None
        coordinates_4 = mul_1 * coordinates_3
        mul_1 = coordinates_3 = None
        sin = torch.sin(coordinates_4)
        cos = torch.cos(coordinates_4)
        coordinates_4 = None
        positional_embedding = torch.cat([sin, cos], dim=-1)
        sin = cos = None
        permute = positional_embedding.permute(2, 0, 1)
        positional_embedding = None
        image_positional_embeddings = permute.unsqueeze(0)
        permute = None
        image_positional_embeddings_1 = image_positional_embeddings.repeat(1, 1, 1, 1)
        image_positional_embeddings = None
        conv2d = torch.conv2d(
            l_kwargs_pixel_values_,
            l_self_modules_vision_encoder_modules_patch_embed_modules_projection_parameters_weight_,
            l_self_modules_vision_encoder_modules_patch_embed_modules_projection_parameters_bias_,
            (16, 16),
            (0, 0),
            (1, 1),
            1,
        )
        l_kwargs_pixel_values_ = l_self_modules_vision_encoder_modules_patch_embed_modules_projection_parameters_weight_ = l_self_modules_vision_encoder_modules_patch_embed_modules_projection_parameters_bias_ = (None)
        embeddings = conv2d.permute(0, 2, 3, 1)
        conv2d = None
        hidden_states = embeddings + l_self_modules_vision_encoder_parameters_pos_embed_
        embeddings = l_self_modules_vision_encoder_parameters_pos_embed_ = None
        item_1 = (
            l_self_modules_vision_encoder_modules_layers_modules_0_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_0_modules_layer_norm1_eps = (
            None
        )
        hidden_states_1 = torch.nn.functional.layer_norm(
            hidden_states,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_,
            item_1,
        )
        l_self_modules_vision_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_ = (item_1) = (
            None
        )
        hidden_states_2 = torch._C._nn.pad(
            hidden_states_1, (0, 0, 0, 6, 0, 6), "constant", None
        )
        hidden_states_1 = None
        hidden_states_3 = hidden_states_2.reshape(1, 5, 14, 5, 14, 1280)
        hidden_states_2 = None
        permute_2 = hidden_states_3.permute(0, 1, 3, 2, 4, 5)
        hidden_states_3 = None
        contiguous = permute_2.contiguous()
        permute_2 = None
        windows = contiguous.reshape(-1, 14, 14, 1280)
        contiguous = None
        linear = torch._C._nn.linear(
            windows,
            l_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        windows = l_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_2 = linear.reshape(25, 196, 3, 16, -1)
        linear = None
        qkv = reshape_2.permute(2, 0, 3, 1, 4)
        reshape_2 = None
        reshape_3 = qkv.reshape(3, 400, 196, -1)
        qkv = None
        unbind = reshape_3.unbind(0)
        reshape_3 = None
        query = unbind[0]
        key = unbind[1]
        value = unbind[2]
        unbind = None
        reshape_4 = l_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_parameters_rel_pos_h_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_4 = reshape_4.permute(0, 2, 1)
        reshape_4 = None
        rel_pos_resized = torch.nn.functional.interpolate(
            permute_4, size=27, mode="linear"
        )
        permute_4 = None
        reshape_5 = rel_pos_resized.reshape(-1, 27)
        rel_pos_resized = None
        rel_pos_resized_1 = reshape_5.permute(1, 0)
        reshape_5 = None
        arange = torch.arange(14)
        getitem_11 = arange[(slice(None, None, None), None)]
        arange = None
        q_coords = getitem_11 * 1.0
        getitem_11 = None
        arange_1 = torch.arange(14)
        getitem_12 = arange_1[(None, slice(None, None, None))]
        arange_1 = None
        k_coords = getitem_12 * 1.0
        getitem_12 = None
        sub_3 = q_coords - k_coords
        q_coords = k_coords = None
        relative_coords = sub_3 + 13.0
        sub_3 = None
        long = relative_coords.long()
        relative_coords = None
        relative_position_height = rel_pos_resized_1[long]
        rel_pos_resized_1 = long = None
        reshape_6 = l_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_parameters_rel_pos_w_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_6 = reshape_6.permute(0, 2, 1)
        reshape_6 = None
        rel_pos_resized_2 = torch.nn.functional.interpolate(
            permute_6, size=27, mode="linear"
        )
        permute_6 = None
        reshape_7 = rel_pos_resized_2.reshape(-1, 27)
        rel_pos_resized_2 = None
        rel_pos_resized_3 = reshape_7.permute(1, 0)
        reshape_7 = None
        arange_2 = torch.arange(14)
        getitem_14 = arange_2[(slice(None, None, None), None)]
        arange_2 = None
        q_coords_1 = getitem_14 * 1.0
        getitem_14 = None
        arange_3 = torch.arange(14)
        getitem_15 = arange_3[(None, slice(None, None, None))]
        arange_3 = None
        k_coords_1 = getitem_15 * 1.0
        getitem_15 = None
        sub_4 = q_coords_1 - k_coords_1
        q_coords_1 = k_coords_1 = None
        relative_coords_1 = sub_4 + 13.0
        sub_4 = None
        long_1 = relative_coords_1.long()
        relative_coords_1 = None
        relative_position_width = rel_pos_resized_3[long_1]
        rel_pos_resized_3 = long_1 = None
        reshaped_query = query.reshape(400, 14, 14, 80)
        rel_h = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query, relative_position_height
        )
        relative_position_height = None
        rel_w = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query, relative_position_width
        )
        reshaped_query = relative_position_width = None
        getitem_17 = rel_h[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h = None
        getitem_18 = rel_w[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w = None
        decomposed_rel_pos = getitem_17 + getitem_18
        getitem_17 = getitem_18 = None
        decomposed_rel_pos_1 = decomposed_rel_pos.reshape(25, 16, 196, 196)
        decomposed_rel_pos = None
        query_1 = query.view(25, 16, 196, -1)
        query = None
        key_1 = key.view(25, 16, 196, -1)
        key = None
        value_1 = value.view(25, 16, 196, -1)
        value = None
        attn_output = torch._C._nn.scaled_dot_product_attention(
            query_1, key_1, value_1, attn_mask=decomposed_rel_pos_1
        )
        query_1 = key_1 = value_1 = decomposed_rel_pos_1 = None
        view_3 = attn_output.view(25, 16, 14, 14, -1)
        attn_output = None
        permute_8 = view_3.permute(0, 2, 3, 1, 4)
        view_3 = None
        attn_output_1 = permute_8.reshape(25, 14, 14, -1)
        permute_8 = None
        attn_output_2 = torch._C._nn.linear(
            attn_output_1,
            l_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_1 = l_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_4 = attn_output_2.reshape(1, 5, 5, 14, 14, -1)
        attn_output_2 = None
        permute_9 = hidden_states_4.permute(0, 1, 3, 2, 4, 5)
        hidden_states_4 = None
        contiguous_1 = permute_9.contiguous()
        permute_9 = None
        hidden_states_5 = contiguous_1.reshape(1, 70, 70, -1)
        contiguous_1 = None
        getitem_19 = hidden_states_5[
            (
                slice(None, None, None),
                slice(None, 64, None),
                slice(None, 64, None),
                slice(None, None, None),
            )
        ]
        hidden_states_5 = None
        hidden_states_6 = getitem_19.contiguous()
        getitem_19 = None
        hidden_states_7 = hidden_states + hidden_states_6
        hidden_states = hidden_states_6 = None
        item_2 = (
            l_self_modules_vision_encoder_modules_layers_modules_0_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_0_modules_layer_norm2_eps = (
            None
        )
        layernorm_output = torch.nn.functional.layer_norm(
            hidden_states_7,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_,
            item_2,
        )
        l_self_modules_vision_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_ = (item_2) = (
            None
        )
        hidden_states_8 = torch._C._nn.linear(
            layernorm_output,
            l_self_modules_vision_encoder_modules_layers_modules_0_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_0_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output = l_self_modules_vision_encoder_modules_layers_modules_0_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_0_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_9 = torch._C._nn.gelu(hidden_states_8)
        hidden_states_8 = None
        hidden_states_10 = torch._C._nn.linear(
            hidden_states_9,
            l_self_modules_vision_encoder_modules_layers_modules_0_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_0_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_9 = l_self_modules_vision_encoder_modules_layers_modules_0_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_0_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_11 = hidden_states_7 + hidden_states_10
        hidden_states_7 = hidden_states_10 = None
        item_3 = (
            l_self_modules_vision_encoder_modules_layers_modules_1_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_1_modules_layer_norm1_eps = (
            None
        )
        hidden_states_12 = torch.nn.functional.layer_norm(
            hidden_states_11,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_bias_,
            item_3,
        )
        l_self_modules_vision_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_bias_ = (item_3) = (
            None
        )
        hidden_states_13 = torch._C._nn.pad(
            hidden_states_12, (0, 0, 0, 6, 0, 6), "constant", None
        )
        hidden_states_12 = None
        hidden_states_14 = hidden_states_13.reshape(1, 5, 14, 5, 14, 1280)
        hidden_states_13 = None
        permute_10 = hidden_states_14.permute(0, 1, 3, 2, 4, 5)
        hidden_states_14 = None
        contiguous_3 = permute_10.contiguous()
        permute_10 = None
        windows_1 = contiguous_3.reshape(-1, 14, 14, 1280)
        contiguous_3 = None
        linear_4 = torch._C._nn.linear(
            windows_1,
            l_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_modules_qkv_parameters_bias_,
        )
        windows_1 = l_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_15 = linear_4.reshape(25, 196, 3, 16, -1)
        linear_4 = None
        qkv_1 = reshape_15.permute(2, 0, 3, 1, 4)
        reshape_15 = None
        reshape_16 = qkv_1.reshape(3, 400, 196, -1)
        qkv_1 = None
        unbind_1 = reshape_16.unbind(0)
        reshape_16 = None
        query_2 = unbind_1[0]
        key_2 = unbind_1[1]
        value_2 = unbind_1[2]
        unbind_1 = None
        reshape_17 = l_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_parameters_rel_pos_h_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_12 = reshape_17.permute(0, 2, 1)
        reshape_17 = None
        rel_pos_resized_4 = torch.nn.functional.interpolate(
            permute_12, size=27, mode="linear"
        )
        permute_12 = None
        reshape_18 = rel_pos_resized_4.reshape(-1, 27)
        rel_pos_resized_4 = None
        rel_pos_resized_5 = reshape_18.permute(1, 0)
        reshape_18 = None
        arange_4 = torch.arange(14)
        getitem_23 = arange_4[(slice(None, None, None), None)]
        arange_4 = None
        q_coords_2 = getitem_23 * 1.0
        getitem_23 = None
        arange_5 = torch.arange(14)
        getitem_24 = arange_5[(None, slice(None, None, None))]
        arange_5 = None
        k_coords_2 = getitem_24 * 1.0
        getitem_24 = None
        sub_5 = q_coords_2 - k_coords_2
        q_coords_2 = k_coords_2 = None
        relative_coords_2 = sub_5 + 13.0
        sub_5 = None
        long_2 = relative_coords_2.long()
        relative_coords_2 = None
        relative_position_height_1 = rel_pos_resized_5[long_2]
        rel_pos_resized_5 = long_2 = None
        reshape_19 = l_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_parameters_rel_pos_w_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_14 = reshape_19.permute(0, 2, 1)
        reshape_19 = None
        rel_pos_resized_6 = torch.nn.functional.interpolate(
            permute_14, size=27, mode="linear"
        )
        permute_14 = None
        reshape_20 = rel_pos_resized_6.reshape(-1, 27)
        rel_pos_resized_6 = None
        rel_pos_resized_7 = reshape_20.permute(1, 0)
        reshape_20 = None
        arange_6 = torch.arange(14)
        getitem_26 = arange_6[(slice(None, None, None), None)]
        arange_6 = None
        q_coords_3 = getitem_26 * 1.0
        getitem_26 = None
        arange_7 = torch.arange(14)
        getitem_27 = arange_7[(None, slice(None, None, None))]
        arange_7 = None
        k_coords_3 = getitem_27 * 1.0
        getitem_27 = None
        sub_6 = q_coords_3 - k_coords_3
        q_coords_3 = k_coords_3 = None
        relative_coords_3 = sub_6 + 13.0
        sub_6 = None
        long_3 = relative_coords_3.long()
        relative_coords_3 = None
        relative_position_width_1 = rel_pos_resized_7[long_3]
        rel_pos_resized_7 = long_3 = None
        reshaped_query_1 = query_2.reshape(400, 14, 14, 80)
        rel_h_1 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_1, relative_position_height_1
        )
        relative_position_height_1 = None
        rel_w_1 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_1, relative_position_width_1
        )
        reshaped_query_1 = relative_position_width_1 = None
        getitem_29 = rel_h_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_1 = None
        getitem_30 = rel_w_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_1 = None
        decomposed_rel_pos_2 = getitem_29 + getitem_30
        getitem_29 = getitem_30 = None
        decomposed_rel_pos_3 = decomposed_rel_pos_2.reshape(25, 16, 196, 196)
        decomposed_rel_pos_2 = None
        query_3 = query_2.view(25, 16, 196, -1)
        query_2 = None
        key_3 = key_2.view(25, 16, 196, -1)
        key_2 = None
        value_3 = value_2.view(25, 16, 196, -1)
        value_2 = None
        attn_output_3 = torch._C._nn.scaled_dot_product_attention(
            query_3, key_3, value_3, attn_mask=decomposed_rel_pos_3
        )
        query_3 = key_3 = value_3 = decomposed_rel_pos_3 = None
        view_7 = attn_output_3.view(25, 16, 14, 14, -1)
        attn_output_3 = None
        permute_16 = view_7.permute(0, 2, 3, 1, 4)
        view_7 = None
        attn_output_4 = permute_16.reshape(25, 14, 14, -1)
        permute_16 = None
        attn_output_5 = torch._C._nn.linear(
            attn_output_4,
            l_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_4 = l_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_15 = attn_output_5.reshape(1, 5, 5, 14, 14, -1)
        attn_output_5 = None
        permute_17 = hidden_states_15.permute(0, 1, 3, 2, 4, 5)
        hidden_states_15 = None
        contiguous_4 = permute_17.contiguous()
        permute_17 = None
        hidden_states_16 = contiguous_4.reshape(1, 70, 70, -1)
        contiguous_4 = None
        getitem_31 = hidden_states_16[
            (
                slice(None, None, None),
                slice(None, 64, None),
                slice(None, 64, None),
                slice(None, None, None),
            )
        ]
        hidden_states_16 = None
        hidden_states_17 = getitem_31.contiguous()
        getitem_31 = None
        hidden_states_18 = hidden_states_11 + hidden_states_17
        hidden_states_11 = hidden_states_17 = None
        item_4 = (
            l_self_modules_vision_encoder_modules_layers_modules_1_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_1_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_1 = torch.nn.functional.layer_norm(
            hidden_states_18,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_bias_,
            item_4,
        )
        l_self_modules_vision_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_bias_ = (item_4) = (
            None
        )
        hidden_states_19 = torch._C._nn.linear(
            layernorm_output_1,
            l_self_modules_vision_encoder_modules_layers_modules_1_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_1_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_1 = l_self_modules_vision_encoder_modules_layers_modules_1_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_1_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_20 = torch._C._nn.gelu(hidden_states_19)
        hidden_states_19 = None
        hidden_states_21 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_vision_encoder_modules_layers_modules_1_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_1_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_20 = l_self_modules_vision_encoder_modules_layers_modules_1_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_1_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_22 = hidden_states_18 + hidden_states_21
        hidden_states_18 = hidden_states_21 = None
        item_5 = (
            l_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm1_eps = (
            None
        )
        hidden_states_23 = torch.nn.functional.layer_norm(
            hidden_states_22,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_bias_,
            item_5,
        )
        l_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_bias_ = (item_5) = (
            None
        )
        hidden_states_24 = torch._C._nn.pad(
            hidden_states_23, (0, 0, 0, 6, 0, 6), "constant", None
        )
        hidden_states_23 = None
        hidden_states_25 = hidden_states_24.reshape(1, 5, 14, 5, 14, 1280)
        hidden_states_24 = None
        permute_18 = hidden_states_25.permute(0, 1, 3, 2, 4, 5)
        hidden_states_25 = None
        contiguous_6 = permute_18.contiguous()
        permute_18 = None
        windows_2 = contiguous_6.reshape(-1, 14, 14, 1280)
        contiguous_6 = None
        linear_8 = torch._C._nn.linear(
            windows_2,
            l_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_modules_qkv_parameters_bias_,
        )
        windows_2 = l_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_28 = linear_8.reshape(25, 196, 3, 16, -1)
        linear_8 = None
        qkv_2 = reshape_28.permute(2, 0, 3, 1, 4)
        reshape_28 = None
        reshape_29 = qkv_2.reshape(3, 400, 196, -1)
        qkv_2 = None
        unbind_2 = reshape_29.unbind(0)
        reshape_29 = None
        query_4 = unbind_2[0]
        key_4 = unbind_2[1]
        value_4 = unbind_2[2]
        unbind_2 = None
        reshape_30 = l_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_parameters_rel_pos_h_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_20 = reshape_30.permute(0, 2, 1)
        reshape_30 = None
        rel_pos_resized_8 = torch.nn.functional.interpolate(
            permute_20, size=27, mode="linear"
        )
        permute_20 = None
        reshape_31 = rel_pos_resized_8.reshape(-1, 27)
        rel_pos_resized_8 = None
        rel_pos_resized_9 = reshape_31.permute(1, 0)
        reshape_31 = None
        arange_8 = torch.arange(14)
        getitem_35 = arange_8[(slice(None, None, None), None)]
        arange_8 = None
        q_coords_4 = getitem_35 * 1.0
        getitem_35 = None
        arange_9 = torch.arange(14)
        getitem_36 = arange_9[(None, slice(None, None, None))]
        arange_9 = None
        k_coords_4 = getitem_36 * 1.0
        getitem_36 = None
        sub_7 = q_coords_4 - k_coords_4
        q_coords_4 = k_coords_4 = None
        relative_coords_4 = sub_7 + 13.0
        sub_7 = None
        long_4 = relative_coords_4.long()
        relative_coords_4 = None
        relative_position_height_2 = rel_pos_resized_9[long_4]
        rel_pos_resized_9 = long_4 = None
        reshape_32 = l_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_parameters_rel_pos_w_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_22 = reshape_32.permute(0, 2, 1)
        reshape_32 = None
        rel_pos_resized_10 = torch.nn.functional.interpolate(
            permute_22, size=27, mode="linear"
        )
        permute_22 = None
        reshape_33 = rel_pos_resized_10.reshape(-1, 27)
        rel_pos_resized_10 = None
        rel_pos_resized_11 = reshape_33.permute(1, 0)
        reshape_33 = None
        arange_10 = torch.arange(14)
        getitem_38 = arange_10[(slice(None, None, None), None)]
        arange_10 = None
        q_coords_5 = getitem_38 * 1.0
        getitem_38 = None
        arange_11 = torch.arange(14)
        getitem_39 = arange_11[(None, slice(None, None, None))]
        arange_11 = None
        k_coords_5 = getitem_39 * 1.0
        getitem_39 = None
        sub_8 = q_coords_5 - k_coords_5
        q_coords_5 = k_coords_5 = None
        relative_coords_5 = sub_8 + 13.0
        sub_8 = None
        long_5 = relative_coords_5.long()
        relative_coords_5 = None
        relative_position_width_2 = rel_pos_resized_11[long_5]
        rel_pos_resized_11 = long_5 = None
        reshaped_query_2 = query_4.reshape(400, 14, 14, 80)
        rel_h_2 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_2, relative_position_height_2
        )
        relative_position_height_2 = None
        rel_w_2 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_2, relative_position_width_2
        )
        reshaped_query_2 = relative_position_width_2 = None
        getitem_41 = rel_h_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_2 = None
        getitem_42 = rel_w_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_2 = None
        decomposed_rel_pos_4 = getitem_41 + getitem_42
        getitem_41 = getitem_42 = None
        decomposed_rel_pos_5 = decomposed_rel_pos_4.reshape(25, 16, 196, 196)
        decomposed_rel_pos_4 = None
        query_5 = query_4.view(25, 16, 196, -1)
        query_4 = None
        key_5 = key_4.view(25, 16, 196, -1)
        key_4 = None
        value_5 = value_4.view(25, 16, 196, -1)
        value_4 = None
        attn_output_6 = torch._C._nn.scaled_dot_product_attention(
            query_5, key_5, value_5, attn_mask=decomposed_rel_pos_5
        )
        query_5 = key_5 = value_5 = decomposed_rel_pos_5 = None
        view_11 = attn_output_6.view(25, 16, 14, 14, -1)
        attn_output_6 = None
        permute_24 = view_11.permute(0, 2, 3, 1, 4)
        view_11 = None
        attn_output_7 = permute_24.reshape(25, 14, 14, -1)
        permute_24 = None
        attn_output_8 = torch._C._nn.linear(
            attn_output_7,
            l_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_7 = l_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_26 = attn_output_8.reshape(1, 5, 5, 14, 14, -1)
        attn_output_8 = None
        permute_25 = hidden_states_26.permute(0, 1, 3, 2, 4, 5)
        hidden_states_26 = None
        contiguous_7 = permute_25.contiguous()
        permute_25 = None
        hidden_states_27 = contiguous_7.reshape(1, 70, 70, -1)
        contiguous_7 = None
        getitem_43 = hidden_states_27[
            (
                slice(None, None, None),
                slice(None, 64, None),
                slice(None, 64, None),
                slice(None, None, None),
            )
        ]
        hidden_states_27 = None
        hidden_states_28 = getitem_43.contiguous()
        getitem_43 = None
        hidden_states_29 = hidden_states_22 + hidden_states_28
        hidden_states_22 = hidden_states_28 = None
        item_6 = (
            l_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_2 = torch.nn.functional.layer_norm(
            hidden_states_29,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_bias_,
            item_6,
        )
        l_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_bias_ = (item_6) = (
            None
        )
        hidden_states_30 = torch._C._nn.linear(
            layernorm_output_2,
            l_self_modules_vision_encoder_modules_layers_modules_2_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_2_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_2 = l_self_modules_vision_encoder_modules_layers_modules_2_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_2_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_31 = torch._C._nn.gelu(hidden_states_30)
        hidden_states_30 = None
        hidden_states_32 = torch._C._nn.linear(
            hidden_states_31,
            l_self_modules_vision_encoder_modules_layers_modules_2_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_2_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_31 = l_self_modules_vision_encoder_modules_layers_modules_2_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_2_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_33 = hidden_states_29 + hidden_states_32
        hidden_states_29 = hidden_states_32 = None
        item_7 = (
            l_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm1_eps = (
            None
        )
        hidden_states_34 = torch.nn.functional.layer_norm(
            hidden_states_33,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_bias_,
            item_7,
        )
        l_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_bias_ = (item_7) = (
            None
        )
        hidden_states_35 = torch._C._nn.pad(
            hidden_states_34, (0, 0, 0, 6, 0, 6), "constant", None
        )
        hidden_states_34 = None
        hidden_states_36 = hidden_states_35.reshape(1, 5, 14, 5, 14, 1280)
        hidden_states_35 = None
        permute_26 = hidden_states_36.permute(0, 1, 3, 2, 4, 5)
        hidden_states_36 = None
        contiguous_9 = permute_26.contiguous()
        permute_26 = None
        windows_3 = contiguous_9.reshape(-1, 14, 14, 1280)
        contiguous_9 = None
        linear_12 = torch._C._nn.linear(
            windows_3,
            l_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_modules_qkv_parameters_bias_,
        )
        windows_3 = l_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_41 = linear_12.reshape(25, 196, 3, 16, -1)
        linear_12 = None
        qkv_3 = reshape_41.permute(2, 0, 3, 1, 4)
        reshape_41 = None
        reshape_42 = qkv_3.reshape(3, 400, 196, -1)
        qkv_3 = None
        unbind_3 = reshape_42.unbind(0)
        reshape_42 = None
        query_6 = unbind_3[0]
        key_6 = unbind_3[1]
        value_6 = unbind_3[2]
        unbind_3 = None
        reshape_43 = l_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_parameters_rel_pos_h_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_28 = reshape_43.permute(0, 2, 1)
        reshape_43 = None
        rel_pos_resized_12 = torch.nn.functional.interpolate(
            permute_28, size=27, mode="linear"
        )
        permute_28 = None
        reshape_44 = rel_pos_resized_12.reshape(-1, 27)
        rel_pos_resized_12 = None
        rel_pos_resized_13 = reshape_44.permute(1, 0)
        reshape_44 = None
        arange_12 = torch.arange(14)
        getitem_47 = arange_12[(slice(None, None, None), None)]
        arange_12 = None
        q_coords_6 = getitem_47 * 1.0
        getitem_47 = None
        arange_13 = torch.arange(14)
        getitem_48 = arange_13[(None, slice(None, None, None))]
        arange_13 = None
        k_coords_6 = getitem_48 * 1.0
        getitem_48 = None
        sub_9 = q_coords_6 - k_coords_6
        q_coords_6 = k_coords_6 = None
        relative_coords_6 = sub_9 + 13.0
        sub_9 = None
        long_6 = relative_coords_6.long()
        relative_coords_6 = None
        relative_position_height_3 = rel_pos_resized_13[long_6]
        rel_pos_resized_13 = long_6 = None
        reshape_45 = l_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_parameters_rel_pos_w_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_30 = reshape_45.permute(0, 2, 1)
        reshape_45 = None
        rel_pos_resized_14 = torch.nn.functional.interpolate(
            permute_30, size=27, mode="linear"
        )
        permute_30 = None
        reshape_46 = rel_pos_resized_14.reshape(-1, 27)
        rel_pos_resized_14 = None
        rel_pos_resized_15 = reshape_46.permute(1, 0)
        reshape_46 = None
        arange_14 = torch.arange(14)
        getitem_50 = arange_14[(slice(None, None, None), None)]
        arange_14 = None
        q_coords_7 = getitem_50 * 1.0
        getitem_50 = None
        arange_15 = torch.arange(14)
        getitem_51 = arange_15[(None, slice(None, None, None))]
        arange_15 = None
        k_coords_7 = getitem_51 * 1.0
        getitem_51 = None
        sub_10 = q_coords_7 - k_coords_7
        q_coords_7 = k_coords_7 = None
        relative_coords_7 = sub_10 + 13.0
        sub_10 = None
        long_7 = relative_coords_7.long()
        relative_coords_7 = None
        relative_position_width_3 = rel_pos_resized_15[long_7]
        rel_pos_resized_15 = long_7 = None
        reshaped_query_3 = query_6.reshape(400, 14, 14, 80)
        rel_h_3 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_3, relative_position_height_3
        )
        relative_position_height_3 = None
        rel_w_3 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_3, relative_position_width_3
        )
        reshaped_query_3 = relative_position_width_3 = None
        getitem_53 = rel_h_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_3 = None
        getitem_54 = rel_w_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_3 = None
        decomposed_rel_pos_6 = getitem_53 + getitem_54
        getitem_53 = getitem_54 = None
        decomposed_rel_pos_7 = decomposed_rel_pos_6.reshape(25, 16, 196, 196)
        decomposed_rel_pos_6 = None
        query_7 = query_6.view(25, 16, 196, -1)
        query_6 = None
        key_7 = key_6.view(25, 16, 196, -1)
        key_6 = None
        value_7 = value_6.view(25, 16, 196, -1)
        value_6 = None
        attn_output_9 = torch._C._nn.scaled_dot_product_attention(
            query_7, key_7, value_7, attn_mask=decomposed_rel_pos_7
        )
        query_7 = key_7 = value_7 = decomposed_rel_pos_7 = None
        view_15 = attn_output_9.view(25, 16, 14, 14, -1)
        attn_output_9 = None
        permute_32 = view_15.permute(0, 2, 3, 1, 4)
        view_15 = None
        attn_output_10 = permute_32.reshape(25, 14, 14, -1)
        permute_32 = None
        attn_output_11 = torch._C._nn.linear(
            attn_output_10,
            l_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_10 = l_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_37 = attn_output_11.reshape(1, 5, 5, 14, 14, -1)
        attn_output_11 = None
        permute_33 = hidden_states_37.permute(0, 1, 3, 2, 4, 5)
        hidden_states_37 = None
        contiguous_10 = permute_33.contiguous()
        permute_33 = None
        hidden_states_38 = contiguous_10.reshape(1, 70, 70, -1)
        contiguous_10 = None
        getitem_55 = hidden_states_38[
            (
                slice(None, None, None),
                slice(None, 64, None),
                slice(None, 64, None),
                slice(None, None, None),
            )
        ]
        hidden_states_38 = None
        hidden_states_39 = getitem_55.contiguous()
        getitem_55 = None
        hidden_states_40 = hidden_states_33 + hidden_states_39
        hidden_states_33 = hidden_states_39 = None
        item_8 = (
            l_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_3 = torch.nn.functional.layer_norm(
            hidden_states_40,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_bias_,
            item_8,
        )
        l_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_bias_ = (item_8) = (
            None
        )
        hidden_states_41 = torch._C._nn.linear(
            layernorm_output_3,
            l_self_modules_vision_encoder_modules_layers_modules_3_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_3_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_3 = l_self_modules_vision_encoder_modules_layers_modules_3_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_3_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_42 = torch._C._nn.gelu(hidden_states_41)
        hidden_states_41 = None
        hidden_states_43 = torch._C._nn.linear(
            hidden_states_42,
            l_self_modules_vision_encoder_modules_layers_modules_3_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_3_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_42 = l_self_modules_vision_encoder_modules_layers_modules_3_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_3_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_44 = hidden_states_40 + hidden_states_43
        hidden_states_40 = hidden_states_43 = None
        item_9 = (
            l_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm1_eps = (
            None
        )
        hidden_states_45 = torch.nn.functional.layer_norm(
            hidden_states_44,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_bias_,
            item_9,
        )
        l_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_bias_ = (item_9) = (
            None
        )
        hidden_states_46 = torch._C._nn.pad(
            hidden_states_45, (0, 0, 0, 6, 0, 6), "constant", None
        )
        hidden_states_45 = None
        hidden_states_47 = hidden_states_46.reshape(1, 5, 14, 5, 14, 1280)
        hidden_states_46 = None
        permute_34 = hidden_states_47.permute(0, 1, 3, 2, 4, 5)
        hidden_states_47 = None
        contiguous_12 = permute_34.contiguous()
        permute_34 = None
        windows_4 = contiguous_12.reshape(-1, 14, 14, 1280)
        contiguous_12 = None
        linear_16 = torch._C._nn.linear(
            windows_4,
            l_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_modules_qkv_parameters_bias_,
        )
        windows_4 = l_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_54 = linear_16.reshape(25, 196, 3, 16, -1)
        linear_16 = None
        qkv_4 = reshape_54.permute(2, 0, 3, 1, 4)
        reshape_54 = None
        reshape_55 = qkv_4.reshape(3, 400, 196, -1)
        qkv_4 = None
        unbind_4 = reshape_55.unbind(0)
        reshape_55 = None
        query_8 = unbind_4[0]
        key_8 = unbind_4[1]
        value_8 = unbind_4[2]
        unbind_4 = None
        reshape_56 = l_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_parameters_rel_pos_h_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_36 = reshape_56.permute(0, 2, 1)
        reshape_56 = None
        rel_pos_resized_16 = torch.nn.functional.interpolate(
            permute_36, size=27, mode="linear"
        )
        permute_36 = None
        reshape_57 = rel_pos_resized_16.reshape(-1, 27)
        rel_pos_resized_16 = None
        rel_pos_resized_17 = reshape_57.permute(1, 0)
        reshape_57 = None
        arange_16 = torch.arange(14)
        getitem_59 = arange_16[(slice(None, None, None), None)]
        arange_16 = None
        q_coords_8 = getitem_59 * 1.0
        getitem_59 = None
        arange_17 = torch.arange(14)
        getitem_60 = arange_17[(None, slice(None, None, None))]
        arange_17 = None
        k_coords_8 = getitem_60 * 1.0
        getitem_60 = None
        sub_11 = q_coords_8 - k_coords_8
        q_coords_8 = k_coords_8 = None
        relative_coords_8 = sub_11 + 13.0
        sub_11 = None
        long_8 = relative_coords_8.long()
        relative_coords_8 = None
        relative_position_height_4 = rel_pos_resized_17[long_8]
        rel_pos_resized_17 = long_8 = None
        reshape_58 = l_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_parameters_rel_pos_w_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_38 = reshape_58.permute(0, 2, 1)
        reshape_58 = None
        rel_pos_resized_18 = torch.nn.functional.interpolate(
            permute_38, size=27, mode="linear"
        )
        permute_38 = None
        reshape_59 = rel_pos_resized_18.reshape(-1, 27)
        rel_pos_resized_18 = None
        rel_pos_resized_19 = reshape_59.permute(1, 0)
        reshape_59 = None
        arange_18 = torch.arange(14)
        getitem_62 = arange_18[(slice(None, None, None), None)]
        arange_18 = None
        q_coords_9 = getitem_62 * 1.0
        getitem_62 = None
        arange_19 = torch.arange(14)
        getitem_63 = arange_19[(None, slice(None, None, None))]
        arange_19 = None
        k_coords_9 = getitem_63 * 1.0
        getitem_63 = None
        sub_12 = q_coords_9 - k_coords_9
        q_coords_9 = k_coords_9 = None
        relative_coords_9 = sub_12 + 13.0
        sub_12 = None
        long_9 = relative_coords_9.long()
        relative_coords_9 = None
        relative_position_width_4 = rel_pos_resized_19[long_9]
        rel_pos_resized_19 = long_9 = None
        reshaped_query_4 = query_8.reshape(400, 14, 14, 80)
        rel_h_4 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_4, relative_position_height_4
        )
        relative_position_height_4 = None
        rel_w_4 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_4, relative_position_width_4
        )
        reshaped_query_4 = relative_position_width_4 = None
        getitem_65 = rel_h_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_4 = None
        getitem_66 = rel_w_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_4 = None
        decomposed_rel_pos_8 = getitem_65 + getitem_66
        getitem_65 = getitem_66 = None
        decomposed_rel_pos_9 = decomposed_rel_pos_8.reshape(25, 16, 196, 196)
        decomposed_rel_pos_8 = None
        query_9 = query_8.view(25, 16, 196, -1)
        query_8 = None
        key_9 = key_8.view(25, 16, 196, -1)
        key_8 = None
        value_9 = value_8.view(25, 16, 196, -1)
        value_8 = None
        attn_output_12 = torch._C._nn.scaled_dot_product_attention(
            query_9, key_9, value_9, attn_mask=decomposed_rel_pos_9
        )
        query_9 = key_9 = value_9 = decomposed_rel_pos_9 = None
        view_19 = attn_output_12.view(25, 16, 14, 14, -1)
        attn_output_12 = None
        permute_40 = view_19.permute(0, 2, 3, 1, 4)
        view_19 = None
        attn_output_13 = permute_40.reshape(25, 14, 14, -1)
        permute_40 = None
        attn_output_14 = torch._C._nn.linear(
            attn_output_13,
            l_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_13 = l_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_48 = attn_output_14.reshape(1, 5, 5, 14, 14, -1)
        attn_output_14 = None
        permute_41 = hidden_states_48.permute(0, 1, 3, 2, 4, 5)
        hidden_states_48 = None
        contiguous_13 = permute_41.contiguous()
        permute_41 = None
        hidden_states_49 = contiguous_13.reshape(1, 70, 70, -1)
        contiguous_13 = None
        getitem_67 = hidden_states_49[
            (
                slice(None, None, None),
                slice(None, 64, None),
                slice(None, 64, None),
                slice(None, None, None),
            )
        ]
        hidden_states_49 = None
        hidden_states_50 = getitem_67.contiguous()
        getitem_67 = None
        hidden_states_51 = hidden_states_44 + hidden_states_50
        hidden_states_44 = hidden_states_50 = None
        item_10 = (
            l_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_4 = torch.nn.functional.layer_norm(
            hidden_states_51,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_bias_,
            item_10,
        )
        l_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_bias_ = (item_10) = (
            None
        )
        hidden_states_52 = torch._C._nn.linear(
            layernorm_output_4,
            l_self_modules_vision_encoder_modules_layers_modules_4_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_4_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_4 = l_self_modules_vision_encoder_modules_layers_modules_4_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_4_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_53 = torch._C._nn.gelu(hidden_states_52)
        hidden_states_52 = None
        hidden_states_54 = torch._C._nn.linear(
            hidden_states_53,
            l_self_modules_vision_encoder_modules_layers_modules_4_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_4_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_53 = l_self_modules_vision_encoder_modules_layers_modules_4_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_4_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_55 = hidden_states_51 + hidden_states_54
        hidden_states_51 = hidden_states_54 = None
        item_11 = (
            l_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm1_eps = (
            None
        )
        hidden_states_56 = torch.nn.functional.layer_norm(
            hidden_states_55,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_bias_,
            item_11,
        )
        l_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_bias_ = (item_11) = (
            None
        )
        hidden_states_57 = torch._C._nn.pad(
            hidden_states_56, (0, 0, 0, 6, 0, 6), "constant", None
        )
        hidden_states_56 = None
        hidden_states_58 = hidden_states_57.reshape(1, 5, 14, 5, 14, 1280)
        hidden_states_57 = None
        permute_42 = hidden_states_58.permute(0, 1, 3, 2, 4, 5)
        hidden_states_58 = None
        contiguous_15 = permute_42.contiguous()
        permute_42 = None
        windows_5 = contiguous_15.reshape(-1, 14, 14, 1280)
        contiguous_15 = None
        linear_20 = torch._C._nn.linear(
            windows_5,
            l_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_modules_qkv_parameters_bias_,
        )
        windows_5 = l_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_67 = linear_20.reshape(25, 196, 3, 16, -1)
        linear_20 = None
        qkv_5 = reshape_67.permute(2, 0, 3, 1, 4)
        reshape_67 = None
        reshape_68 = qkv_5.reshape(3, 400, 196, -1)
        qkv_5 = None
        unbind_5 = reshape_68.unbind(0)
        reshape_68 = None
        query_10 = unbind_5[0]
        key_10 = unbind_5[1]
        value_10 = unbind_5[2]
        unbind_5 = None
        reshape_69 = l_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_parameters_rel_pos_h_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_44 = reshape_69.permute(0, 2, 1)
        reshape_69 = None
        rel_pos_resized_20 = torch.nn.functional.interpolate(
            permute_44, size=27, mode="linear"
        )
        permute_44 = None
        reshape_70 = rel_pos_resized_20.reshape(-1, 27)
        rel_pos_resized_20 = None
        rel_pos_resized_21 = reshape_70.permute(1, 0)
        reshape_70 = None
        arange_20 = torch.arange(14)
        getitem_71 = arange_20[(slice(None, None, None), None)]
        arange_20 = None
        q_coords_10 = getitem_71 * 1.0
        getitem_71 = None
        arange_21 = torch.arange(14)
        getitem_72 = arange_21[(None, slice(None, None, None))]
        arange_21 = None
        k_coords_10 = getitem_72 * 1.0
        getitem_72 = None
        sub_13 = q_coords_10 - k_coords_10
        q_coords_10 = k_coords_10 = None
        relative_coords_10 = sub_13 + 13.0
        sub_13 = None
        long_10 = relative_coords_10.long()
        relative_coords_10 = None
        relative_position_height_5 = rel_pos_resized_21[long_10]
        rel_pos_resized_21 = long_10 = None
        reshape_71 = l_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_parameters_rel_pos_w_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_46 = reshape_71.permute(0, 2, 1)
        reshape_71 = None
        rel_pos_resized_22 = torch.nn.functional.interpolate(
            permute_46, size=27, mode="linear"
        )
        permute_46 = None
        reshape_72 = rel_pos_resized_22.reshape(-1, 27)
        rel_pos_resized_22 = None
        rel_pos_resized_23 = reshape_72.permute(1, 0)
        reshape_72 = None
        arange_22 = torch.arange(14)
        getitem_74 = arange_22[(slice(None, None, None), None)]
        arange_22 = None
        q_coords_11 = getitem_74 * 1.0
        getitem_74 = None
        arange_23 = torch.arange(14)
        getitem_75 = arange_23[(None, slice(None, None, None))]
        arange_23 = None
        k_coords_11 = getitem_75 * 1.0
        getitem_75 = None
        sub_14 = q_coords_11 - k_coords_11
        q_coords_11 = k_coords_11 = None
        relative_coords_11 = sub_14 + 13.0
        sub_14 = None
        long_11 = relative_coords_11.long()
        relative_coords_11 = None
        relative_position_width_5 = rel_pos_resized_23[long_11]
        rel_pos_resized_23 = long_11 = None
        reshaped_query_5 = query_10.reshape(400, 14, 14, 80)
        rel_h_5 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_5, relative_position_height_5
        )
        relative_position_height_5 = None
        rel_w_5 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_5, relative_position_width_5
        )
        reshaped_query_5 = relative_position_width_5 = None
        getitem_77 = rel_h_5[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_5 = None
        getitem_78 = rel_w_5[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_5 = None
        decomposed_rel_pos_10 = getitem_77 + getitem_78
        getitem_77 = getitem_78 = None
        decomposed_rel_pos_11 = decomposed_rel_pos_10.reshape(25, 16, 196, 196)
        decomposed_rel_pos_10 = None
        query_11 = query_10.view(25, 16, 196, -1)
        query_10 = None
        key_11 = key_10.view(25, 16, 196, -1)
        key_10 = None
        value_11 = value_10.view(25, 16, 196, -1)
        value_10 = None
        attn_output_15 = torch._C._nn.scaled_dot_product_attention(
            query_11, key_11, value_11, attn_mask=decomposed_rel_pos_11
        )
        query_11 = key_11 = value_11 = decomposed_rel_pos_11 = None
        view_23 = attn_output_15.view(25, 16, 14, 14, -1)
        attn_output_15 = None
        permute_48 = view_23.permute(0, 2, 3, 1, 4)
        view_23 = None
        attn_output_16 = permute_48.reshape(25, 14, 14, -1)
        permute_48 = None
        attn_output_17 = torch._C._nn.linear(
            attn_output_16,
            l_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_16 = l_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_59 = attn_output_17.reshape(1, 5, 5, 14, 14, -1)
        attn_output_17 = None
        permute_49 = hidden_states_59.permute(0, 1, 3, 2, 4, 5)
        hidden_states_59 = None
        contiguous_16 = permute_49.contiguous()
        permute_49 = None
        hidden_states_60 = contiguous_16.reshape(1, 70, 70, -1)
        contiguous_16 = None
        getitem_79 = hidden_states_60[
            (
                slice(None, None, None),
                slice(None, 64, None),
                slice(None, 64, None),
                slice(None, None, None),
            )
        ]
        hidden_states_60 = None
        hidden_states_61 = getitem_79.contiguous()
        getitem_79 = None
        hidden_states_62 = hidden_states_55 + hidden_states_61
        hidden_states_55 = hidden_states_61 = None
        item_12 = (
            l_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_5 = torch.nn.functional.layer_norm(
            hidden_states_62,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_bias_,
            item_12,
        )
        l_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_bias_ = (item_12) = (
            None
        )
        hidden_states_63 = torch._C._nn.linear(
            layernorm_output_5,
            l_self_modules_vision_encoder_modules_layers_modules_5_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_5_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_5 = l_self_modules_vision_encoder_modules_layers_modules_5_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_5_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_64 = torch._C._nn.gelu(hidden_states_63)
        hidden_states_63 = None
        hidden_states_65 = torch._C._nn.linear(
            hidden_states_64,
            l_self_modules_vision_encoder_modules_layers_modules_5_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_5_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_64 = l_self_modules_vision_encoder_modules_layers_modules_5_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_5_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_66 = hidden_states_62 + hidden_states_65
        hidden_states_62 = hidden_states_65 = None
        item_13 = (
            l_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm1_eps = (
            None
        )
        hidden_states_67 = torch.nn.functional.layer_norm(
            hidden_states_66,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_bias_,
            item_13,
        )
        l_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_bias_ = (item_13) = (
            None
        )
        hidden_states_68 = torch._C._nn.pad(
            hidden_states_67, (0, 0, 0, 6, 0, 6), "constant", None
        )
        hidden_states_67 = None
        hidden_states_69 = hidden_states_68.reshape(1, 5, 14, 5, 14, 1280)
        hidden_states_68 = None
        permute_50 = hidden_states_69.permute(0, 1, 3, 2, 4, 5)
        hidden_states_69 = None
        contiguous_18 = permute_50.contiguous()
        permute_50 = None
        windows_6 = contiguous_18.reshape(-1, 14, 14, 1280)
        contiguous_18 = None
        linear_24 = torch._C._nn.linear(
            windows_6,
            l_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_modules_qkv_parameters_bias_,
        )
        windows_6 = l_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_80 = linear_24.reshape(25, 196, 3, 16, -1)
        linear_24 = None
        qkv_6 = reshape_80.permute(2, 0, 3, 1, 4)
        reshape_80 = None
        reshape_81 = qkv_6.reshape(3, 400, 196, -1)
        qkv_6 = None
        unbind_6 = reshape_81.unbind(0)
        reshape_81 = None
        query_12 = unbind_6[0]
        key_12 = unbind_6[1]
        value_12 = unbind_6[2]
        unbind_6 = None
        reshape_82 = l_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_parameters_rel_pos_h_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_52 = reshape_82.permute(0, 2, 1)
        reshape_82 = None
        rel_pos_resized_24 = torch.nn.functional.interpolate(
            permute_52, size=27, mode="linear"
        )
        permute_52 = None
        reshape_83 = rel_pos_resized_24.reshape(-1, 27)
        rel_pos_resized_24 = None
        rel_pos_resized_25 = reshape_83.permute(1, 0)
        reshape_83 = None
        arange_24 = torch.arange(14)
        getitem_83 = arange_24[(slice(None, None, None), None)]
        arange_24 = None
        q_coords_12 = getitem_83 * 1.0
        getitem_83 = None
        arange_25 = torch.arange(14)
        getitem_84 = arange_25[(None, slice(None, None, None))]
        arange_25 = None
        k_coords_12 = getitem_84 * 1.0
        getitem_84 = None
        sub_15 = q_coords_12 - k_coords_12
        q_coords_12 = k_coords_12 = None
        relative_coords_12 = sub_15 + 13.0
        sub_15 = None
        long_12 = relative_coords_12.long()
        relative_coords_12 = None
        relative_position_height_6 = rel_pos_resized_25[long_12]
        rel_pos_resized_25 = long_12 = None
        reshape_84 = l_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_parameters_rel_pos_w_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_54 = reshape_84.permute(0, 2, 1)
        reshape_84 = None
        rel_pos_resized_26 = torch.nn.functional.interpolate(
            permute_54, size=27, mode="linear"
        )
        permute_54 = None
        reshape_85 = rel_pos_resized_26.reshape(-1, 27)
        rel_pos_resized_26 = None
        rel_pos_resized_27 = reshape_85.permute(1, 0)
        reshape_85 = None
        arange_26 = torch.arange(14)
        getitem_86 = arange_26[(slice(None, None, None), None)]
        arange_26 = None
        q_coords_13 = getitem_86 * 1.0
        getitem_86 = None
        arange_27 = torch.arange(14)
        getitem_87 = arange_27[(None, slice(None, None, None))]
        arange_27 = None
        k_coords_13 = getitem_87 * 1.0
        getitem_87 = None
        sub_16 = q_coords_13 - k_coords_13
        q_coords_13 = k_coords_13 = None
        relative_coords_13 = sub_16 + 13.0
        sub_16 = None
        long_13 = relative_coords_13.long()
        relative_coords_13 = None
        relative_position_width_6 = rel_pos_resized_27[long_13]
        rel_pos_resized_27 = long_13 = None
        reshaped_query_6 = query_12.reshape(400, 14, 14, 80)
        rel_h_6 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_6, relative_position_height_6
        )
        relative_position_height_6 = None
        rel_w_6 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_6, relative_position_width_6
        )
        reshaped_query_6 = relative_position_width_6 = None
        getitem_89 = rel_h_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_6 = None
        getitem_90 = rel_w_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_6 = None
        decomposed_rel_pos_12 = getitem_89 + getitem_90
        getitem_89 = getitem_90 = None
        decomposed_rel_pos_13 = decomposed_rel_pos_12.reshape(25, 16, 196, 196)
        decomposed_rel_pos_12 = None
        query_13 = query_12.view(25, 16, 196, -1)
        query_12 = None
        key_13 = key_12.view(25, 16, 196, -1)
        key_12 = None
        value_13 = value_12.view(25, 16, 196, -1)
        value_12 = None
        attn_output_18 = torch._C._nn.scaled_dot_product_attention(
            query_13, key_13, value_13, attn_mask=decomposed_rel_pos_13
        )
        query_13 = key_13 = value_13 = decomposed_rel_pos_13 = None
        view_27 = attn_output_18.view(25, 16, 14, 14, -1)
        attn_output_18 = None
        permute_56 = view_27.permute(0, 2, 3, 1, 4)
        view_27 = None
        attn_output_19 = permute_56.reshape(25, 14, 14, -1)
        permute_56 = None
        attn_output_20 = torch._C._nn.linear(
            attn_output_19,
            l_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_19 = l_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_70 = attn_output_20.reshape(1, 5, 5, 14, 14, -1)
        attn_output_20 = None
        permute_57 = hidden_states_70.permute(0, 1, 3, 2, 4, 5)
        hidden_states_70 = None
        contiguous_19 = permute_57.contiguous()
        permute_57 = None
        hidden_states_71 = contiguous_19.reshape(1, 70, 70, -1)
        contiguous_19 = None
        getitem_91 = hidden_states_71[
            (
                slice(None, None, None),
                slice(None, 64, None),
                slice(None, 64, None),
                slice(None, None, None),
            )
        ]
        hidden_states_71 = None
        hidden_states_72 = getitem_91.contiguous()
        getitem_91 = None
        hidden_states_73 = hidden_states_66 + hidden_states_72
        hidden_states_66 = hidden_states_72 = None
        item_14 = (
            l_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_6 = torch.nn.functional.layer_norm(
            hidden_states_73,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_bias_,
            item_14,
        )
        l_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_bias_ = (item_14) = (
            None
        )
        hidden_states_74 = torch._C._nn.linear(
            layernorm_output_6,
            l_self_modules_vision_encoder_modules_layers_modules_6_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_6_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_6 = l_self_modules_vision_encoder_modules_layers_modules_6_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_6_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_75 = torch._C._nn.gelu(hidden_states_74)
        hidden_states_74 = None
        hidden_states_76 = torch._C._nn.linear(
            hidden_states_75,
            l_self_modules_vision_encoder_modules_layers_modules_6_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_6_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_75 = l_self_modules_vision_encoder_modules_layers_modules_6_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_6_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_77 = hidden_states_73 + hidden_states_76
        hidden_states_73 = hidden_states_76 = None
        item_15 = (
            l_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm1_eps = (
            None
        )
        hidden_states_78 = torch.nn.functional.layer_norm(
            hidden_states_77,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_bias_,
            item_15,
        )
        l_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_bias_ = (item_15) = (
            None
        )
        linear_28 = torch._C._nn.linear(
            hidden_states_78,
            l_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_modules_qkv_parameters_bias_,
        )
        hidden_states_78 = l_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_91 = linear_28.reshape(1, 4096, 3, 16, -1)
        linear_28 = None
        qkv_7 = reshape_91.permute(2, 0, 3, 1, 4)
        reshape_91 = None
        reshape_92 = qkv_7.reshape(3, 16, 4096, -1)
        qkv_7 = None
        unbind_7 = reshape_92.unbind(0)
        reshape_92 = None
        query_14 = unbind_7[0]
        key_14 = unbind_7[1]
        value_14 = unbind_7[2]
        unbind_7 = None
        reshape_93 = l_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_parameters_rel_pos_h_.reshape(
            1, 127, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_59 = reshape_93.permute(0, 2, 1)
        reshape_93 = None
        rel_pos_resized_28 = torch.nn.functional.interpolate(
            permute_59, size=127, mode="linear"
        )
        permute_59 = None
        reshape_94 = rel_pos_resized_28.reshape(-1, 127)
        rel_pos_resized_28 = None
        rel_pos_resized_29 = reshape_94.permute(1, 0)
        reshape_94 = None
        arange_28 = torch.arange(64)
        getitem_95 = arange_28[(slice(None, None, None), None)]
        arange_28 = None
        q_coords_14 = getitem_95 * 1.0
        getitem_95 = None
        arange_29 = torch.arange(64)
        getitem_96 = arange_29[(None, slice(None, None, None))]
        arange_29 = None
        k_coords_14 = getitem_96 * 1.0
        getitem_96 = None
        sub_17 = q_coords_14 - k_coords_14
        q_coords_14 = k_coords_14 = None
        relative_coords_14 = sub_17 + 63.0
        sub_17 = None
        long_14 = relative_coords_14.long()
        relative_coords_14 = None
        relative_position_height_7 = rel_pos_resized_29[long_14]
        rel_pos_resized_29 = long_14 = None
        reshape_95 = l_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_parameters_rel_pos_w_.reshape(
            1, 127, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_61 = reshape_95.permute(0, 2, 1)
        reshape_95 = None
        rel_pos_resized_30 = torch.nn.functional.interpolate(
            permute_61, size=127, mode="linear"
        )
        permute_61 = None
        reshape_96 = rel_pos_resized_30.reshape(-1, 127)
        rel_pos_resized_30 = None
        rel_pos_resized_31 = reshape_96.permute(1, 0)
        reshape_96 = None
        arange_30 = torch.arange(64)
        getitem_98 = arange_30[(slice(None, None, None), None)]
        arange_30 = None
        q_coords_15 = getitem_98 * 1.0
        getitem_98 = None
        arange_31 = torch.arange(64)
        getitem_99 = arange_31[(None, slice(None, None, None))]
        arange_31 = None
        k_coords_15 = getitem_99 * 1.0
        getitem_99 = None
        sub_18 = q_coords_15 - k_coords_15
        q_coords_15 = k_coords_15 = None
        relative_coords_15 = sub_18 + 63.0
        sub_18 = None
        long_15 = relative_coords_15.long()
        relative_coords_15 = None
        relative_position_width_7 = rel_pos_resized_31[long_15]
        rel_pos_resized_31 = long_15 = None
        reshaped_query_7 = query_14.reshape(16, 64, 64, 80)
        rel_h_7 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_7, relative_position_height_7
        )
        relative_position_height_7 = None
        rel_w_7 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_7, relative_position_width_7
        )
        reshaped_query_7 = relative_position_width_7 = None
        getitem_101 = rel_h_7[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_7 = None
        getitem_102 = rel_w_7[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_7 = None
        decomposed_rel_pos_14 = getitem_101 + getitem_102
        getitem_101 = getitem_102 = None
        decomposed_rel_pos_15 = decomposed_rel_pos_14.reshape(1, 16, 4096, 4096)
        decomposed_rel_pos_14 = None
        query_15 = query_14.view(1, 16, 4096, -1)
        query_14 = None
        key_15 = key_14.view(1, 16, 4096, -1)
        key_14 = None
        value_15 = value_14.view(1, 16, 4096, -1)
        value_14 = None
        attn_output_21 = torch._C._nn.scaled_dot_product_attention(
            query_15, key_15, value_15, attn_mask=decomposed_rel_pos_15
        )
        query_15 = key_15 = value_15 = decomposed_rel_pos_15 = None
        view_31 = attn_output_21.view(1, 16, 64, 64, -1)
        attn_output_21 = None
        permute_63 = view_31.permute(0, 2, 3, 1, 4)
        view_31 = None
        attn_output_22 = permute_63.reshape(1, 64, 64, -1)
        permute_63 = None
        attn_output_23 = torch._C._nn.linear(
            attn_output_22,
            l_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_22 = l_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_79 = hidden_states_77 + attn_output_23
        hidden_states_77 = attn_output_23 = None
        item_16 = (
            l_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_7 = torch.nn.functional.layer_norm(
            hidden_states_79,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_bias_,
            item_16,
        )
        l_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_bias_ = (item_16) = (
            None
        )
        hidden_states_80 = torch._C._nn.linear(
            layernorm_output_7,
            l_self_modules_vision_encoder_modules_layers_modules_7_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_7_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_7 = l_self_modules_vision_encoder_modules_layers_modules_7_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_7_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_81 = torch._C._nn.gelu(hidden_states_80)
        hidden_states_80 = None
        hidden_states_82 = torch._C._nn.linear(
            hidden_states_81,
            l_self_modules_vision_encoder_modules_layers_modules_7_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_7_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_81 = l_self_modules_vision_encoder_modules_layers_modules_7_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_7_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_83 = hidden_states_79 + hidden_states_82
        hidden_states_79 = hidden_states_82 = None
        item_17 = (
            l_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm1_eps = (
            None
        )
        hidden_states_84 = torch.nn.functional.layer_norm(
            hidden_states_83,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_bias_,
            item_17,
        )
        l_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_bias_ = (item_17) = (
            None
        )
        hidden_states_85 = torch._C._nn.pad(
            hidden_states_84, (0, 0, 0, 6, 0, 6), "constant", None
        )
        hidden_states_84 = None
        hidden_states_86 = hidden_states_85.reshape(1, 5, 14, 5, 14, 1280)
        hidden_states_85 = None
        permute_64 = hidden_states_86.permute(0, 1, 3, 2, 4, 5)
        hidden_states_86 = None
        contiguous_21 = permute_64.contiguous()
        permute_64 = None
        windows_7 = contiguous_21.reshape(-1, 14, 14, 1280)
        contiguous_21 = None
        linear_32 = torch._C._nn.linear(
            windows_7,
            l_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_modules_qkv_parameters_bias_,
        )
        windows_7 = l_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_102 = linear_32.reshape(25, 196, 3, 16, -1)
        linear_32 = None
        qkv_8 = reshape_102.permute(2, 0, 3, 1, 4)
        reshape_102 = None
        reshape_103 = qkv_8.reshape(3, 400, 196, -1)
        qkv_8 = None
        unbind_8 = reshape_103.unbind(0)
        reshape_103 = None
        query_16 = unbind_8[0]
        key_16 = unbind_8[1]
        value_16 = unbind_8[2]
        unbind_8 = None
        reshape_104 = l_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_parameters_rel_pos_h_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_66 = reshape_104.permute(0, 2, 1)
        reshape_104 = None
        rel_pos_resized_32 = torch.nn.functional.interpolate(
            permute_66, size=27, mode="linear"
        )
        permute_66 = None
        reshape_105 = rel_pos_resized_32.reshape(-1, 27)
        rel_pos_resized_32 = None
        rel_pos_resized_33 = reshape_105.permute(1, 0)
        reshape_105 = None
        arange_32 = torch.arange(14)
        getitem_106 = arange_32[(slice(None, None, None), None)]
        arange_32 = None
        q_coords_16 = getitem_106 * 1.0
        getitem_106 = None
        arange_33 = torch.arange(14)
        getitem_107 = arange_33[(None, slice(None, None, None))]
        arange_33 = None
        k_coords_16 = getitem_107 * 1.0
        getitem_107 = None
        sub_19 = q_coords_16 - k_coords_16
        q_coords_16 = k_coords_16 = None
        relative_coords_16 = sub_19 + 13.0
        sub_19 = None
        long_16 = relative_coords_16.long()
        relative_coords_16 = None
        relative_position_height_8 = rel_pos_resized_33[long_16]
        rel_pos_resized_33 = long_16 = None
        reshape_106 = l_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_parameters_rel_pos_w_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_68 = reshape_106.permute(0, 2, 1)
        reshape_106 = None
        rel_pos_resized_34 = torch.nn.functional.interpolate(
            permute_68, size=27, mode="linear"
        )
        permute_68 = None
        reshape_107 = rel_pos_resized_34.reshape(-1, 27)
        rel_pos_resized_34 = None
        rel_pos_resized_35 = reshape_107.permute(1, 0)
        reshape_107 = None
        arange_34 = torch.arange(14)
        getitem_109 = arange_34[(slice(None, None, None), None)]
        arange_34 = None
        q_coords_17 = getitem_109 * 1.0
        getitem_109 = None
        arange_35 = torch.arange(14)
        getitem_110 = arange_35[(None, slice(None, None, None))]
        arange_35 = None
        k_coords_17 = getitem_110 * 1.0
        getitem_110 = None
        sub_20 = q_coords_17 - k_coords_17
        q_coords_17 = k_coords_17 = None
        relative_coords_17 = sub_20 + 13.0
        sub_20 = None
        long_17 = relative_coords_17.long()
        relative_coords_17 = None
        relative_position_width_8 = rel_pos_resized_35[long_17]
        rel_pos_resized_35 = long_17 = None
        reshaped_query_8 = query_16.reshape(400, 14, 14, 80)
        rel_h_8 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_8, relative_position_height_8
        )
        relative_position_height_8 = None
        rel_w_8 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_8, relative_position_width_8
        )
        reshaped_query_8 = relative_position_width_8 = None
        getitem_112 = rel_h_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_8 = None
        getitem_113 = rel_w_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_8 = None
        decomposed_rel_pos_16 = getitem_112 + getitem_113
        getitem_112 = getitem_113 = None
        decomposed_rel_pos_17 = decomposed_rel_pos_16.reshape(25, 16, 196, 196)
        decomposed_rel_pos_16 = None
        query_17 = query_16.view(25, 16, 196, -1)
        query_16 = None
        key_17 = key_16.view(25, 16, 196, -1)
        key_16 = None
        value_17 = value_16.view(25, 16, 196, -1)
        value_16 = None
        attn_output_24 = torch._C._nn.scaled_dot_product_attention(
            query_17, key_17, value_17, attn_mask=decomposed_rel_pos_17
        )
        query_17 = key_17 = value_17 = decomposed_rel_pos_17 = None
        view_35 = attn_output_24.view(25, 16, 14, 14, -1)
        attn_output_24 = None
        permute_70 = view_35.permute(0, 2, 3, 1, 4)
        view_35 = None
        attn_output_25 = permute_70.reshape(25, 14, 14, -1)
        permute_70 = None
        attn_output_26 = torch._C._nn.linear(
            attn_output_25,
            l_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_25 = l_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_87 = attn_output_26.reshape(1, 5, 5, 14, 14, -1)
        attn_output_26 = None
        permute_71 = hidden_states_87.permute(0, 1, 3, 2, 4, 5)
        hidden_states_87 = None
        contiguous_22 = permute_71.contiguous()
        permute_71 = None
        hidden_states_88 = contiguous_22.reshape(1, 70, 70, -1)
        contiguous_22 = None
        getitem_114 = hidden_states_88[
            (
                slice(None, None, None),
                slice(None, 64, None),
                slice(None, 64, None),
                slice(None, None, None),
            )
        ]
        hidden_states_88 = None
        hidden_states_89 = getitem_114.contiguous()
        getitem_114 = None
        hidden_states_90 = hidden_states_83 + hidden_states_89
        hidden_states_83 = hidden_states_89 = None
        item_18 = (
            l_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_8 = torch.nn.functional.layer_norm(
            hidden_states_90,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_bias_,
            item_18,
        )
        l_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_bias_ = (item_18) = (
            None
        )
        hidden_states_91 = torch._C._nn.linear(
            layernorm_output_8,
            l_self_modules_vision_encoder_modules_layers_modules_8_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_8_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_8 = l_self_modules_vision_encoder_modules_layers_modules_8_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_8_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_92 = torch._C._nn.gelu(hidden_states_91)
        hidden_states_91 = None
        hidden_states_93 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_vision_encoder_modules_layers_modules_8_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_8_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_92 = l_self_modules_vision_encoder_modules_layers_modules_8_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_8_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_94 = hidden_states_90 + hidden_states_93
        hidden_states_90 = hidden_states_93 = None
        item_19 = (
            l_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm1_eps = (
            None
        )
        hidden_states_95 = torch.nn.functional.layer_norm(
            hidden_states_94,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_bias_,
            item_19,
        )
        l_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_bias_ = (item_19) = (
            None
        )
        hidden_states_96 = torch._C._nn.pad(
            hidden_states_95, (0, 0, 0, 6, 0, 6), "constant", None
        )
        hidden_states_95 = None
        hidden_states_97 = hidden_states_96.reshape(1, 5, 14, 5, 14, 1280)
        hidden_states_96 = None
        permute_72 = hidden_states_97.permute(0, 1, 3, 2, 4, 5)
        hidden_states_97 = None
        contiguous_24 = permute_72.contiguous()
        permute_72 = None
        windows_8 = contiguous_24.reshape(-1, 14, 14, 1280)
        contiguous_24 = None
        linear_36 = torch._C._nn.linear(
            windows_8,
            l_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_modules_qkv_parameters_bias_,
        )
        windows_8 = l_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_115 = linear_36.reshape(25, 196, 3, 16, -1)
        linear_36 = None
        qkv_9 = reshape_115.permute(2, 0, 3, 1, 4)
        reshape_115 = None
        reshape_116 = qkv_9.reshape(3, 400, 196, -1)
        qkv_9 = None
        unbind_9 = reshape_116.unbind(0)
        reshape_116 = None
        query_18 = unbind_9[0]
        key_18 = unbind_9[1]
        value_18 = unbind_9[2]
        unbind_9 = None
        reshape_117 = l_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_parameters_rel_pos_h_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_74 = reshape_117.permute(0, 2, 1)
        reshape_117 = None
        rel_pos_resized_36 = torch.nn.functional.interpolate(
            permute_74, size=27, mode="linear"
        )
        permute_74 = None
        reshape_118 = rel_pos_resized_36.reshape(-1, 27)
        rel_pos_resized_36 = None
        rel_pos_resized_37 = reshape_118.permute(1, 0)
        reshape_118 = None
        arange_36 = torch.arange(14)
        getitem_118 = arange_36[(slice(None, None, None), None)]
        arange_36 = None
        q_coords_18 = getitem_118 * 1.0
        getitem_118 = None
        arange_37 = torch.arange(14)
        getitem_119 = arange_37[(None, slice(None, None, None))]
        arange_37 = None
        k_coords_18 = getitem_119 * 1.0
        getitem_119 = None
        sub_21 = q_coords_18 - k_coords_18
        q_coords_18 = k_coords_18 = None
        relative_coords_18 = sub_21 + 13.0
        sub_21 = None
        long_18 = relative_coords_18.long()
        relative_coords_18 = None
        relative_position_height_9 = rel_pos_resized_37[long_18]
        rel_pos_resized_37 = long_18 = None
        reshape_119 = l_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_parameters_rel_pos_w_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_76 = reshape_119.permute(0, 2, 1)
        reshape_119 = None
        rel_pos_resized_38 = torch.nn.functional.interpolate(
            permute_76, size=27, mode="linear"
        )
        permute_76 = None
        reshape_120 = rel_pos_resized_38.reshape(-1, 27)
        rel_pos_resized_38 = None
        rel_pos_resized_39 = reshape_120.permute(1, 0)
        reshape_120 = None
        arange_38 = torch.arange(14)
        getitem_121 = arange_38[(slice(None, None, None), None)]
        arange_38 = None
        q_coords_19 = getitem_121 * 1.0
        getitem_121 = None
        arange_39 = torch.arange(14)
        getitem_122 = arange_39[(None, slice(None, None, None))]
        arange_39 = None
        k_coords_19 = getitem_122 * 1.0
        getitem_122 = None
        sub_22 = q_coords_19 - k_coords_19
        q_coords_19 = k_coords_19 = None
        relative_coords_19 = sub_22 + 13.0
        sub_22 = None
        long_19 = relative_coords_19.long()
        relative_coords_19 = None
        relative_position_width_9 = rel_pos_resized_39[long_19]
        rel_pos_resized_39 = long_19 = None
        reshaped_query_9 = query_18.reshape(400, 14, 14, 80)
        rel_h_9 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_9, relative_position_height_9
        )
        relative_position_height_9 = None
        rel_w_9 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_9, relative_position_width_9
        )
        reshaped_query_9 = relative_position_width_9 = None
        getitem_124 = rel_h_9[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_9 = None
        getitem_125 = rel_w_9[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_9 = None
        decomposed_rel_pos_18 = getitem_124 + getitem_125
        getitem_124 = getitem_125 = None
        decomposed_rel_pos_19 = decomposed_rel_pos_18.reshape(25, 16, 196, 196)
        decomposed_rel_pos_18 = None
        query_19 = query_18.view(25, 16, 196, -1)
        query_18 = None
        key_19 = key_18.view(25, 16, 196, -1)
        key_18 = None
        value_19 = value_18.view(25, 16, 196, -1)
        value_18 = None
        attn_output_27 = torch._C._nn.scaled_dot_product_attention(
            query_19, key_19, value_19, attn_mask=decomposed_rel_pos_19
        )
        query_19 = key_19 = value_19 = decomposed_rel_pos_19 = None
        view_39 = attn_output_27.view(25, 16, 14, 14, -1)
        attn_output_27 = None
        permute_78 = view_39.permute(0, 2, 3, 1, 4)
        view_39 = None
        attn_output_28 = permute_78.reshape(25, 14, 14, -1)
        permute_78 = None
        attn_output_29 = torch._C._nn.linear(
            attn_output_28,
            l_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_28 = l_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_98 = attn_output_29.reshape(1, 5, 5, 14, 14, -1)
        attn_output_29 = None
        permute_79 = hidden_states_98.permute(0, 1, 3, 2, 4, 5)
        hidden_states_98 = None
        contiguous_25 = permute_79.contiguous()
        permute_79 = None
        hidden_states_99 = contiguous_25.reshape(1, 70, 70, -1)
        contiguous_25 = None
        getitem_126 = hidden_states_99[
            (
                slice(None, None, None),
                slice(None, 64, None),
                slice(None, 64, None),
                slice(None, None, None),
            )
        ]
        hidden_states_99 = None
        hidden_states_100 = getitem_126.contiguous()
        getitem_126 = None
        hidden_states_101 = hidden_states_94 + hidden_states_100
        hidden_states_94 = hidden_states_100 = None
        item_20 = (
            l_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_9 = torch.nn.functional.layer_norm(
            hidden_states_101,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_bias_,
            item_20,
        )
        l_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_bias_ = (item_20) = (
            None
        )
        hidden_states_102 = torch._C._nn.linear(
            layernorm_output_9,
            l_self_modules_vision_encoder_modules_layers_modules_9_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_9_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_9 = l_self_modules_vision_encoder_modules_layers_modules_9_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_9_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_103 = torch._C._nn.gelu(hidden_states_102)
        hidden_states_102 = None
        hidden_states_104 = torch._C._nn.linear(
            hidden_states_103,
            l_self_modules_vision_encoder_modules_layers_modules_9_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_9_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_103 = l_self_modules_vision_encoder_modules_layers_modules_9_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_9_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_105 = hidden_states_101 + hidden_states_104
        hidden_states_101 = hidden_states_104 = None
        item_21 = (
            l_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm1_eps = (
            None
        )
        hidden_states_106 = torch.nn.functional.layer_norm(
            hidden_states_105,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_bias_,
            item_21,
        )
        l_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_bias_ = (item_21) = (
            None
        )
        hidden_states_107 = torch._C._nn.pad(
            hidden_states_106, (0, 0, 0, 6, 0, 6), "constant", None
        )
        hidden_states_106 = None
        hidden_states_108 = hidden_states_107.reshape(1, 5, 14, 5, 14, 1280)
        hidden_states_107 = None
        permute_80 = hidden_states_108.permute(0, 1, 3, 2, 4, 5)
        hidden_states_108 = None
        contiguous_27 = permute_80.contiguous()
        permute_80 = None
        windows_9 = contiguous_27.reshape(-1, 14, 14, 1280)
        contiguous_27 = None
        linear_40 = torch._C._nn.linear(
            windows_9,
            l_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_modules_qkv_parameters_bias_,
        )
        windows_9 = l_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_128 = linear_40.reshape(25, 196, 3, 16, -1)
        linear_40 = None
        qkv_10 = reshape_128.permute(2, 0, 3, 1, 4)
        reshape_128 = None
        reshape_129 = qkv_10.reshape(3, 400, 196, -1)
        qkv_10 = None
        unbind_10 = reshape_129.unbind(0)
        reshape_129 = None
        query_20 = unbind_10[0]
        key_20 = unbind_10[1]
        value_20 = unbind_10[2]
        unbind_10 = None
        reshape_130 = l_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_parameters_rel_pos_h_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_82 = reshape_130.permute(0, 2, 1)
        reshape_130 = None
        rel_pos_resized_40 = torch.nn.functional.interpolate(
            permute_82, size=27, mode="linear"
        )
        permute_82 = None
        reshape_131 = rel_pos_resized_40.reshape(-1, 27)
        rel_pos_resized_40 = None
        rel_pos_resized_41 = reshape_131.permute(1, 0)
        reshape_131 = None
        arange_40 = torch.arange(14)
        getitem_130 = arange_40[(slice(None, None, None), None)]
        arange_40 = None
        q_coords_20 = getitem_130 * 1.0
        getitem_130 = None
        arange_41 = torch.arange(14)
        getitem_131 = arange_41[(None, slice(None, None, None))]
        arange_41 = None
        k_coords_20 = getitem_131 * 1.0
        getitem_131 = None
        sub_23 = q_coords_20 - k_coords_20
        q_coords_20 = k_coords_20 = None
        relative_coords_20 = sub_23 + 13.0
        sub_23 = None
        long_20 = relative_coords_20.long()
        relative_coords_20 = None
        relative_position_height_10 = rel_pos_resized_41[long_20]
        rel_pos_resized_41 = long_20 = None
        reshape_132 = l_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_parameters_rel_pos_w_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_84 = reshape_132.permute(0, 2, 1)
        reshape_132 = None
        rel_pos_resized_42 = torch.nn.functional.interpolate(
            permute_84, size=27, mode="linear"
        )
        permute_84 = None
        reshape_133 = rel_pos_resized_42.reshape(-1, 27)
        rel_pos_resized_42 = None
        rel_pos_resized_43 = reshape_133.permute(1, 0)
        reshape_133 = None
        arange_42 = torch.arange(14)
        getitem_133 = arange_42[(slice(None, None, None), None)]
        arange_42 = None
        q_coords_21 = getitem_133 * 1.0
        getitem_133 = None
        arange_43 = torch.arange(14)
        getitem_134 = arange_43[(None, slice(None, None, None))]
        arange_43 = None
        k_coords_21 = getitem_134 * 1.0
        getitem_134 = None
        sub_24 = q_coords_21 - k_coords_21
        q_coords_21 = k_coords_21 = None
        relative_coords_21 = sub_24 + 13.0
        sub_24 = None
        long_21 = relative_coords_21.long()
        relative_coords_21 = None
        relative_position_width_10 = rel_pos_resized_43[long_21]
        rel_pos_resized_43 = long_21 = None
        reshaped_query_10 = query_20.reshape(400, 14, 14, 80)
        rel_h_10 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_10, relative_position_height_10
        )
        relative_position_height_10 = None
        rel_w_10 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_10, relative_position_width_10
        )
        reshaped_query_10 = relative_position_width_10 = None
        getitem_136 = rel_h_10[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_10 = None
        getitem_137 = rel_w_10[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_10 = None
        decomposed_rel_pos_20 = getitem_136 + getitem_137
        getitem_136 = getitem_137 = None
        decomposed_rel_pos_21 = decomposed_rel_pos_20.reshape(25, 16, 196, 196)
        decomposed_rel_pos_20 = None
        query_21 = query_20.view(25, 16, 196, -1)
        query_20 = None
        key_21 = key_20.view(25, 16, 196, -1)
        key_20 = None
        value_21 = value_20.view(25, 16, 196, -1)
        value_20 = None
        attn_output_30 = torch._C._nn.scaled_dot_product_attention(
            query_21, key_21, value_21, attn_mask=decomposed_rel_pos_21
        )
        query_21 = key_21 = value_21 = decomposed_rel_pos_21 = None
        view_43 = attn_output_30.view(25, 16, 14, 14, -1)
        attn_output_30 = None
        permute_86 = view_43.permute(0, 2, 3, 1, 4)
        view_43 = None
        attn_output_31 = permute_86.reshape(25, 14, 14, -1)
        permute_86 = None
        attn_output_32 = torch._C._nn.linear(
            attn_output_31,
            l_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_31 = l_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_109 = attn_output_32.reshape(1, 5, 5, 14, 14, -1)
        attn_output_32 = None
        permute_87 = hidden_states_109.permute(0, 1, 3, 2, 4, 5)
        hidden_states_109 = None
        contiguous_28 = permute_87.contiguous()
        permute_87 = None
        hidden_states_110 = contiguous_28.reshape(1, 70, 70, -1)
        contiguous_28 = None
        getitem_138 = hidden_states_110[
            (
                slice(None, None, None),
                slice(None, 64, None),
                slice(None, 64, None),
                slice(None, None, None),
            )
        ]
        hidden_states_110 = None
        hidden_states_111 = getitem_138.contiguous()
        getitem_138 = None
        hidden_states_112 = hidden_states_105 + hidden_states_111
        hidden_states_105 = hidden_states_111 = None
        item_22 = (
            l_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_10 = torch.nn.functional.layer_norm(
            hidden_states_112,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_bias_,
            item_22,
        )
        l_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_bias_ = (item_22) = (
            None
        )
        hidden_states_113 = torch._C._nn.linear(
            layernorm_output_10,
            l_self_modules_vision_encoder_modules_layers_modules_10_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_10_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_10 = l_self_modules_vision_encoder_modules_layers_modules_10_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_10_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_114 = torch._C._nn.gelu(hidden_states_113)
        hidden_states_113 = None
        hidden_states_115 = torch._C._nn.linear(
            hidden_states_114,
            l_self_modules_vision_encoder_modules_layers_modules_10_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_10_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_114 = l_self_modules_vision_encoder_modules_layers_modules_10_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_10_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_116 = hidden_states_112 + hidden_states_115
        hidden_states_112 = hidden_states_115 = None
        item_23 = (
            l_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm1_eps = (
            None
        )
        hidden_states_117 = torch.nn.functional.layer_norm(
            hidden_states_116,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_bias_,
            item_23,
        )
        l_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_bias_ = (item_23) = (
            None
        )
        hidden_states_118 = torch._C._nn.pad(
            hidden_states_117, (0, 0, 0, 6, 0, 6), "constant", None
        )
        hidden_states_117 = None
        hidden_states_119 = hidden_states_118.reshape(1, 5, 14, 5, 14, 1280)
        hidden_states_118 = None
        permute_88 = hidden_states_119.permute(0, 1, 3, 2, 4, 5)
        hidden_states_119 = None
        contiguous_30 = permute_88.contiguous()
        permute_88 = None
        windows_10 = contiguous_30.reshape(-1, 14, 14, 1280)
        contiguous_30 = None
        linear_44 = torch._C._nn.linear(
            windows_10,
            l_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_modules_qkv_parameters_bias_,
        )
        windows_10 = l_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_141 = linear_44.reshape(25, 196, 3, 16, -1)
        linear_44 = None
        qkv_11 = reshape_141.permute(2, 0, 3, 1, 4)
        reshape_141 = None
        reshape_142 = qkv_11.reshape(3, 400, 196, -1)
        qkv_11 = None
        unbind_11 = reshape_142.unbind(0)
        reshape_142 = None
        query_22 = unbind_11[0]
        key_22 = unbind_11[1]
        value_22 = unbind_11[2]
        unbind_11 = None
        reshape_143 = l_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_parameters_rel_pos_h_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_90 = reshape_143.permute(0, 2, 1)
        reshape_143 = None
        rel_pos_resized_44 = torch.nn.functional.interpolate(
            permute_90, size=27, mode="linear"
        )
        permute_90 = None
        reshape_144 = rel_pos_resized_44.reshape(-1, 27)
        rel_pos_resized_44 = None
        rel_pos_resized_45 = reshape_144.permute(1, 0)
        reshape_144 = None
        arange_44 = torch.arange(14)
        getitem_142 = arange_44[(slice(None, None, None), None)]
        arange_44 = None
        q_coords_22 = getitem_142 * 1.0
        getitem_142 = None
        arange_45 = torch.arange(14)
        getitem_143 = arange_45[(None, slice(None, None, None))]
        arange_45 = None
        k_coords_22 = getitem_143 * 1.0
        getitem_143 = None
        sub_25 = q_coords_22 - k_coords_22
        q_coords_22 = k_coords_22 = None
        relative_coords_22 = sub_25 + 13.0
        sub_25 = None
        long_22 = relative_coords_22.long()
        relative_coords_22 = None
        relative_position_height_11 = rel_pos_resized_45[long_22]
        rel_pos_resized_45 = long_22 = None
        reshape_145 = l_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_parameters_rel_pos_w_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_92 = reshape_145.permute(0, 2, 1)
        reshape_145 = None
        rel_pos_resized_46 = torch.nn.functional.interpolate(
            permute_92, size=27, mode="linear"
        )
        permute_92 = None
        reshape_146 = rel_pos_resized_46.reshape(-1, 27)
        rel_pos_resized_46 = None
        rel_pos_resized_47 = reshape_146.permute(1, 0)
        reshape_146 = None
        arange_46 = torch.arange(14)
        getitem_145 = arange_46[(slice(None, None, None), None)]
        arange_46 = None
        q_coords_23 = getitem_145 * 1.0
        getitem_145 = None
        arange_47 = torch.arange(14)
        getitem_146 = arange_47[(None, slice(None, None, None))]
        arange_47 = None
        k_coords_23 = getitem_146 * 1.0
        getitem_146 = None
        sub_26 = q_coords_23 - k_coords_23
        q_coords_23 = k_coords_23 = None
        relative_coords_23 = sub_26 + 13.0
        sub_26 = None
        long_23 = relative_coords_23.long()
        relative_coords_23 = None
        relative_position_width_11 = rel_pos_resized_47[long_23]
        rel_pos_resized_47 = long_23 = None
        reshaped_query_11 = query_22.reshape(400, 14, 14, 80)
        rel_h_11 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_11, relative_position_height_11
        )
        relative_position_height_11 = None
        rel_w_11 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_11, relative_position_width_11
        )
        reshaped_query_11 = relative_position_width_11 = None
        getitem_148 = rel_h_11[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_11 = None
        getitem_149 = rel_w_11[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_11 = None
        decomposed_rel_pos_22 = getitem_148 + getitem_149
        getitem_148 = getitem_149 = None
        decomposed_rel_pos_23 = decomposed_rel_pos_22.reshape(25, 16, 196, 196)
        decomposed_rel_pos_22 = None
        query_23 = query_22.view(25, 16, 196, -1)
        query_22 = None
        key_23 = key_22.view(25, 16, 196, -1)
        key_22 = None
        value_23 = value_22.view(25, 16, 196, -1)
        value_22 = None
        attn_output_33 = torch._C._nn.scaled_dot_product_attention(
            query_23, key_23, value_23, attn_mask=decomposed_rel_pos_23
        )
        query_23 = key_23 = value_23 = decomposed_rel_pos_23 = None
        view_47 = attn_output_33.view(25, 16, 14, 14, -1)
        attn_output_33 = None
        permute_94 = view_47.permute(0, 2, 3, 1, 4)
        view_47 = None
        attn_output_34 = permute_94.reshape(25, 14, 14, -1)
        permute_94 = None
        attn_output_35 = torch._C._nn.linear(
            attn_output_34,
            l_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_34 = l_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_120 = attn_output_35.reshape(1, 5, 5, 14, 14, -1)
        attn_output_35 = None
        permute_95 = hidden_states_120.permute(0, 1, 3, 2, 4, 5)
        hidden_states_120 = None
        contiguous_31 = permute_95.contiguous()
        permute_95 = None
        hidden_states_121 = contiguous_31.reshape(1, 70, 70, -1)
        contiguous_31 = None
        getitem_150 = hidden_states_121[
            (
                slice(None, None, None),
                slice(None, 64, None),
                slice(None, 64, None),
                slice(None, None, None),
            )
        ]
        hidden_states_121 = None
        hidden_states_122 = getitem_150.contiguous()
        getitem_150 = None
        hidden_states_123 = hidden_states_116 + hidden_states_122
        hidden_states_116 = hidden_states_122 = None
        item_24 = (
            l_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_11 = torch.nn.functional.layer_norm(
            hidden_states_123,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_bias_,
            item_24,
        )
        l_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_bias_ = (item_24) = (
            None
        )
        hidden_states_124 = torch._C._nn.linear(
            layernorm_output_11,
            l_self_modules_vision_encoder_modules_layers_modules_11_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_11_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_11 = l_self_modules_vision_encoder_modules_layers_modules_11_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_11_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_125 = torch._C._nn.gelu(hidden_states_124)
        hidden_states_124 = None
        hidden_states_126 = torch._C._nn.linear(
            hidden_states_125,
            l_self_modules_vision_encoder_modules_layers_modules_11_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_11_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_125 = l_self_modules_vision_encoder_modules_layers_modules_11_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_11_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_127 = hidden_states_123 + hidden_states_126
        hidden_states_123 = hidden_states_126 = None
        item_25 = (
            l_self_modules_vision_encoder_modules_layers_modules_12_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_12_modules_layer_norm1_eps = (
            None
        )
        hidden_states_128 = torch.nn.functional.layer_norm(
            hidden_states_127,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_bias_,
            item_25,
        )
        l_self_modules_vision_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_bias_ = (item_25) = (
            None
        )
        hidden_states_129 = torch._C._nn.pad(
            hidden_states_128, (0, 0, 0, 6, 0, 6), "constant", None
        )
        hidden_states_128 = None
        hidden_states_130 = hidden_states_129.reshape(1, 5, 14, 5, 14, 1280)
        hidden_states_129 = None
        permute_96 = hidden_states_130.permute(0, 1, 3, 2, 4, 5)
        hidden_states_130 = None
        contiguous_33 = permute_96.contiguous()
        permute_96 = None
        windows_11 = contiguous_33.reshape(-1, 14, 14, 1280)
        contiguous_33 = None
        linear_48 = torch._C._nn.linear(
            windows_11,
            l_self_modules_vision_encoder_modules_layers_modules_12_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_12_modules_attn_modules_qkv_parameters_bias_,
        )
        windows_11 = l_self_modules_vision_encoder_modules_layers_modules_12_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_12_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_154 = linear_48.reshape(25, 196, 3, 16, -1)
        linear_48 = None
        qkv_12 = reshape_154.permute(2, 0, 3, 1, 4)
        reshape_154 = None
        reshape_155 = qkv_12.reshape(3, 400, 196, -1)
        qkv_12 = None
        unbind_12 = reshape_155.unbind(0)
        reshape_155 = None
        query_24 = unbind_12[0]
        key_24 = unbind_12[1]
        value_24 = unbind_12[2]
        unbind_12 = None
        reshape_156 = l_self_modules_vision_encoder_modules_layers_modules_12_modules_attn_parameters_rel_pos_h_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_12_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_98 = reshape_156.permute(0, 2, 1)
        reshape_156 = None
        rel_pos_resized_48 = torch.nn.functional.interpolate(
            permute_98, size=27, mode="linear"
        )
        permute_98 = None
        reshape_157 = rel_pos_resized_48.reshape(-1, 27)
        rel_pos_resized_48 = None
        rel_pos_resized_49 = reshape_157.permute(1, 0)
        reshape_157 = None
        arange_48 = torch.arange(14)
        getitem_154 = arange_48[(slice(None, None, None), None)]
        arange_48 = None
        q_coords_24 = getitem_154 * 1.0
        getitem_154 = None
        arange_49 = torch.arange(14)
        getitem_155 = arange_49[(None, slice(None, None, None))]
        arange_49 = None
        k_coords_24 = getitem_155 * 1.0
        getitem_155 = None
        sub_27 = q_coords_24 - k_coords_24
        q_coords_24 = k_coords_24 = None
        relative_coords_24 = sub_27 + 13.0
        sub_27 = None
        long_24 = relative_coords_24.long()
        relative_coords_24 = None
        relative_position_height_12 = rel_pos_resized_49[long_24]
        rel_pos_resized_49 = long_24 = None
        reshape_158 = l_self_modules_vision_encoder_modules_layers_modules_12_modules_attn_parameters_rel_pos_w_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_12_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_100 = reshape_158.permute(0, 2, 1)
        reshape_158 = None
        rel_pos_resized_50 = torch.nn.functional.interpolate(
            permute_100, size=27, mode="linear"
        )
        permute_100 = None
        reshape_159 = rel_pos_resized_50.reshape(-1, 27)
        rel_pos_resized_50 = None
        rel_pos_resized_51 = reshape_159.permute(1, 0)
        reshape_159 = None
        arange_50 = torch.arange(14)
        getitem_157 = arange_50[(slice(None, None, None), None)]
        arange_50 = None
        q_coords_25 = getitem_157 * 1.0
        getitem_157 = None
        arange_51 = torch.arange(14)
        getitem_158 = arange_51[(None, slice(None, None, None))]
        arange_51 = None
        k_coords_25 = getitem_158 * 1.0
        getitem_158 = None
        sub_28 = q_coords_25 - k_coords_25
        q_coords_25 = k_coords_25 = None
        relative_coords_25 = sub_28 + 13.0
        sub_28 = None
        long_25 = relative_coords_25.long()
        relative_coords_25 = None
        relative_position_width_12 = rel_pos_resized_51[long_25]
        rel_pos_resized_51 = long_25 = None
        reshaped_query_12 = query_24.reshape(400, 14, 14, 80)
        rel_h_12 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_12, relative_position_height_12
        )
        relative_position_height_12 = None
        rel_w_12 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_12, relative_position_width_12
        )
        reshaped_query_12 = relative_position_width_12 = None
        getitem_160 = rel_h_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_12 = None
        getitem_161 = rel_w_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_12 = None
        decomposed_rel_pos_24 = getitem_160 + getitem_161
        getitem_160 = getitem_161 = None
        decomposed_rel_pos_25 = decomposed_rel_pos_24.reshape(25, 16, 196, 196)
        decomposed_rel_pos_24 = None
        query_25 = query_24.view(25, 16, 196, -1)
        query_24 = None
        key_25 = key_24.view(25, 16, 196, -1)
        key_24 = None
        value_25 = value_24.view(25, 16, 196, -1)
        value_24 = None
        attn_output_36 = torch._C._nn.scaled_dot_product_attention(
            query_25, key_25, value_25, attn_mask=decomposed_rel_pos_25
        )
        query_25 = key_25 = value_25 = decomposed_rel_pos_25 = None
        view_51 = attn_output_36.view(25, 16, 14, 14, -1)
        attn_output_36 = None
        permute_102 = view_51.permute(0, 2, 3, 1, 4)
        view_51 = None
        attn_output_37 = permute_102.reshape(25, 14, 14, -1)
        permute_102 = None
        attn_output_38 = torch._C._nn.linear(
            attn_output_37,
            l_self_modules_vision_encoder_modules_layers_modules_12_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_12_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_37 = l_self_modules_vision_encoder_modules_layers_modules_12_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_12_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_131 = attn_output_38.reshape(1, 5, 5, 14, 14, -1)
        attn_output_38 = None
        permute_103 = hidden_states_131.permute(0, 1, 3, 2, 4, 5)
        hidden_states_131 = None
        contiguous_34 = permute_103.contiguous()
        permute_103 = None
        hidden_states_132 = contiguous_34.reshape(1, 70, 70, -1)
        contiguous_34 = None
        getitem_162 = hidden_states_132[
            (
                slice(None, None, None),
                slice(None, 64, None),
                slice(None, 64, None),
                slice(None, None, None),
            )
        ]
        hidden_states_132 = None
        hidden_states_133 = getitem_162.contiguous()
        getitem_162 = None
        hidden_states_134 = hidden_states_127 + hidden_states_133
        hidden_states_127 = hidden_states_133 = None
        item_26 = (
            l_self_modules_vision_encoder_modules_layers_modules_12_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_12_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_12 = torch.nn.functional.layer_norm(
            hidden_states_134,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_bias_,
            item_26,
        )
        l_self_modules_vision_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_bias_ = (item_26) = (
            None
        )
        hidden_states_135 = torch._C._nn.linear(
            layernorm_output_12,
            l_self_modules_vision_encoder_modules_layers_modules_12_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_12_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_12 = l_self_modules_vision_encoder_modules_layers_modules_12_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_12_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_136 = torch._C._nn.gelu(hidden_states_135)
        hidden_states_135 = None
        hidden_states_137 = torch._C._nn.linear(
            hidden_states_136,
            l_self_modules_vision_encoder_modules_layers_modules_12_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_12_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_136 = l_self_modules_vision_encoder_modules_layers_modules_12_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_12_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_138 = hidden_states_134 + hidden_states_137
        hidden_states_134 = hidden_states_137 = None
        item_27 = (
            l_self_modules_vision_encoder_modules_layers_modules_13_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_13_modules_layer_norm1_eps = (
            None
        )
        hidden_states_139 = torch.nn.functional.layer_norm(
            hidden_states_138,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_bias_,
            item_27,
        )
        l_self_modules_vision_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_bias_ = (item_27) = (
            None
        )
        hidden_states_140 = torch._C._nn.pad(
            hidden_states_139, (0, 0, 0, 6, 0, 6), "constant", None
        )
        hidden_states_139 = None
        hidden_states_141 = hidden_states_140.reshape(1, 5, 14, 5, 14, 1280)
        hidden_states_140 = None
        permute_104 = hidden_states_141.permute(0, 1, 3, 2, 4, 5)
        hidden_states_141 = None
        contiguous_36 = permute_104.contiguous()
        permute_104 = None
        windows_12 = contiguous_36.reshape(-1, 14, 14, 1280)
        contiguous_36 = None
        linear_52 = torch._C._nn.linear(
            windows_12,
            l_self_modules_vision_encoder_modules_layers_modules_13_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_13_modules_attn_modules_qkv_parameters_bias_,
        )
        windows_12 = l_self_modules_vision_encoder_modules_layers_modules_13_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_13_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_167 = linear_52.reshape(25, 196, 3, 16, -1)
        linear_52 = None
        qkv_13 = reshape_167.permute(2, 0, 3, 1, 4)
        reshape_167 = None
        reshape_168 = qkv_13.reshape(3, 400, 196, -1)
        qkv_13 = None
        unbind_13 = reshape_168.unbind(0)
        reshape_168 = None
        query_26 = unbind_13[0]
        key_26 = unbind_13[1]
        value_26 = unbind_13[2]
        unbind_13 = None
        reshape_169 = l_self_modules_vision_encoder_modules_layers_modules_13_modules_attn_parameters_rel_pos_h_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_13_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_106 = reshape_169.permute(0, 2, 1)
        reshape_169 = None
        rel_pos_resized_52 = torch.nn.functional.interpolate(
            permute_106, size=27, mode="linear"
        )
        permute_106 = None
        reshape_170 = rel_pos_resized_52.reshape(-1, 27)
        rel_pos_resized_52 = None
        rel_pos_resized_53 = reshape_170.permute(1, 0)
        reshape_170 = None
        arange_52 = torch.arange(14)
        getitem_166 = arange_52[(slice(None, None, None), None)]
        arange_52 = None
        q_coords_26 = getitem_166 * 1.0
        getitem_166 = None
        arange_53 = torch.arange(14)
        getitem_167 = arange_53[(None, slice(None, None, None))]
        arange_53 = None
        k_coords_26 = getitem_167 * 1.0
        getitem_167 = None
        sub_29 = q_coords_26 - k_coords_26
        q_coords_26 = k_coords_26 = None
        relative_coords_26 = sub_29 + 13.0
        sub_29 = None
        long_26 = relative_coords_26.long()
        relative_coords_26 = None
        relative_position_height_13 = rel_pos_resized_53[long_26]
        rel_pos_resized_53 = long_26 = None
        reshape_171 = l_self_modules_vision_encoder_modules_layers_modules_13_modules_attn_parameters_rel_pos_w_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_13_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_108 = reshape_171.permute(0, 2, 1)
        reshape_171 = None
        rel_pos_resized_54 = torch.nn.functional.interpolate(
            permute_108, size=27, mode="linear"
        )
        permute_108 = None
        reshape_172 = rel_pos_resized_54.reshape(-1, 27)
        rel_pos_resized_54 = None
        rel_pos_resized_55 = reshape_172.permute(1, 0)
        reshape_172 = None
        arange_54 = torch.arange(14)
        getitem_169 = arange_54[(slice(None, None, None), None)]
        arange_54 = None
        q_coords_27 = getitem_169 * 1.0
        getitem_169 = None
        arange_55 = torch.arange(14)
        getitem_170 = arange_55[(None, slice(None, None, None))]
        arange_55 = None
        k_coords_27 = getitem_170 * 1.0
        getitem_170 = None
        sub_30 = q_coords_27 - k_coords_27
        q_coords_27 = k_coords_27 = None
        relative_coords_27 = sub_30 + 13.0
        sub_30 = None
        long_27 = relative_coords_27.long()
        relative_coords_27 = None
        relative_position_width_13 = rel_pos_resized_55[long_27]
        rel_pos_resized_55 = long_27 = None
        reshaped_query_13 = query_26.reshape(400, 14, 14, 80)
        rel_h_13 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_13, relative_position_height_13
        )
        relative_position_height_13 = None
        rel_w_13 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_13, relative_position_width_13
        )
        reshaped_query_13 = relative_position_width_13 = None
        getitem_172 = rel_h_13[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_13 = None
        getitem_173 = rel_w_13[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_13 = None
        decomposed_rel_pos_26 = getitem_172 + getitem_173
        getitem_172 = getitem_173 = None
        decomposed_rel_pos_27 = decomposed_rel_pos_26.reshape(25, 16, 196, 196)
        decomposed_rel_pos_26 = None
        query_27 = query_26.view(25, 16, 196, -1)
        query_26 = None
        key_27 = key_26.view(25, 16, 196, -1)
        key_26 = None
        value_27 = value_26.view(25, 16, 196, -1)
        value_26 = None
        attn_output_39 = torch._C._nn.scaled_dot_product_attention(
            query_27, key_27, value_27, attn_mask=decomposed_rel_pos_27
        )
        query_27 = key_27 = value_27 = decomposed_rel_pos_27 = None
        view_55 = attn_output_39.view(25, 16, 14, 14, -1)
        attn_output_39 = None
        permute_110 = view_55.permute(0, 2, 3, 1, 4)
        view_55 = None
        attn_output_40 = permute_110.reshape(25, 14, 14, -1)
        permute_110 = None
        attn_output_41 = torch._C._nn.linear(
            attn_output_40,
            l_self_modules_vision_encoder_modules_layers_modules_13_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_13_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_40 = l_self_modules_vision_encoder_modules_layers_modules_13_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_13_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_142 = attn_output_41.reshape(1, 5, 5, 14, 14, -1)
        attn_output_41 = None
        permute_111 = hidden_states_142.permute(0, 1, 3, 2, 4, 5)
        hidden_states_142 = None
        contiguous_37 = permute_111.contiguous()
        permute_111 = None
        hidden_states_143 = contiguous_37.reshape(1, 70, 70, -1)
        contiguous_37 = None
        getitem_174 = hidden_states_143[
            (
                slice(None, None, None),
                slice(None, 64, None),
                slice(None, 64, None),
                slice(None, None, None),
            )
        ]
        hidden_states_143 = None
        hidden_states_144 = getitem_174.contiguous()
        getitem_174 = None
        hidden_states_145 = hidden_states_138 + hidden_states_144
        hidden_states_138 = hidden_states_144 = None
        item_28 = (
            l_self_modules_vision_encoder_modules_layers_modules_13_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_13_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_13 = torch.nn.functional.layer_norm(
            hidden_states_145,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_bias_,
            item_28,
        )
        l_self_modules_vision_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_bias_ = (item_28) = (
            None
        )
        hidden_states_146 = torch._C._nn.linear(
            layernorm_output_13,
            l_self_modules_vision_encoder_modules_layers_modules_13_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_13_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_13 = l_self_modules_vision_encoder_modules_layers_modules_13_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_13_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_147 = torch._C._nn.gelu(hidden_states_146)
        hidden_states_146 = None
        hidden_states_148 = torch._C._nn.linear(
            hidden_states_147,
            l_self_modules_vision_encoder_modules_layers_modules_13_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_13_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_147 = l_self_modules_vision_encoder_modules_layers_modules_13_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_13_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_149 = hidden_states_145 + hidden_states_148
        hidden_states_145 = hidden_states_148 = None
        item_29 = (
            l_self_modules_vision_encoder_modules_layers_modules_14_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_14_modules_layer_norm1_eps = (
            None
        )
        hidden_states_150 = torch.nn.functional.layer_norm(
            hidden_states_149,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_bias_,
            item_29,
        )
        l_self_modules_vision_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_bias_ = (item_29) = (
            None
        )
        hidden_states_151 = torch._C._nn.pad(
            hidden_states_150, (0, 0, 0, 6, 0, 6), "constant", None
        )
        hidden_states_150 = None
        hidden_states_152 = hidden_states_151.reshape(1, 5, 14, 5, 14, 1280)
        hidden_states_151 = None
        permute_112 = hidden_states_152.permute(0, 1, 3, 2, 4, 5)
        hidden_states_152 = None
        contiguous_39 = permute_112.contiguous()
        permute_112 = None
        windows_13 = contiguous_39.reshape(-1, 14, 14, 1280)
        contiguous_39 = None
        linear_56 = torch._C._nn.linear(
            windows_13,
            l_self_modules_vision_encoder_modules_layers_modules_14_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_14_modules_attn_modules_qkv_parameters_bias_,
        )
        windows_13 = l_self_modules_vision_encoder_modules_layers_modules_14_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_14_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_180 = linear_56.reshape(25, 196, 3, 16, -1)
        linear_56 = None
        qkv_14 = reshape_180.permute(2, 0, 3, 1, 4)
        reshape_180 = None
        reshape_181 = qkv_14.reshape(3, 400, 196, -1)
        qkv_14 = None
        unbind_14 = reshape_181.unbind(0)
        reshape_181 = None
        query_28 = unbind_14[0]
        key_28 = unbind_14[1]
        value_28 = unbind_14[2]
        unbind_14 = None
        reshape_182 = l_self_modules_vision_encoder_modules_layers_modules_14_modules_attn_parameters_rel_pos_h_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_14_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_114 = reshape_182.permute(0, 2, 1)
        reshape_182 = None
        rel_pos_resized_56 = torch.nn.functional.interpolate(
            permute_114, size=27, mode="linear"
        )
        permute_114 = None
        reshape_183 = rel_pos_resized_56.reshape(-1, 27)
        rel_pos_resized_56 = None
        rel_pos_resized_57 = reshape_183.permute(1, 0)
        reshape_183 = None
        arange_56 = torch.arange(14)
        getitem_178 = arange_56[(slice(None, None, None), None)]
        arange_56 = None
        q_coords_28 = getitem_178 * 1.0
        getitem_178 = None
        arange_57 = torch.arange(14)
        getitem_179 = arange_57[(None, slice(None, None, None))]
        arange_57 = None
        k_coords_28 = getitem_179 * 1.0
        getitem_179 = None
        sub_31 = q_coords_28 - k_coords_28
        q_coords_28 = k_coords_28 = None
        relative_coords_28 = sub_31 + 13.0
        sub_31 = None
        long_28 = relative_coords_28.long()
        relative_coords_28 = None
        relative_position_height_14 = rel_pos_resized_57[long_28]
        rel_pos_resized_57 = long_28 = None
        reshape_184 = l_self_modules_vision_encoder_modules_layers_modules_14_modules_attn_parameters_rel_pos_w_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_14_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_116 = reshape_184.permute(0, 2, 1)
        reshape_184 = None
        rel_pos_resized_58 = torch.nn.functional.interpolate(
            permute_116, size=27, mode="linear"
        )
        permute_116 = None
        reshape_185 = rel_pos_resized_58.reshape(-1, 27)
        rel_pos_resized_58 = None
        rel_pos_resized_59 = reshape_185.permute(1, 0)
        reshape_185 = None
        arange_58 = torch.arange(14)
        getitem_181 = arange_58[(slice(None, None, None), None)]
        arange_58 = None
        q_coords_29 = getitem_181 * 1.0
        getitem_181 = None
        arange_59 = torch.arange(14)
        getitem_182 = arange_59[(None, slice(None, None, None))]
        arange_59 = None
        k_coords_29 = getitem_182 * 1.0
        getitem_182 = None
        sub_32 = q_coords_29 - k_coords_29
        q_coords_29 = k_coords_29 = None
        relative_coords_29 = sub_32 + 13.0
        sub_32 = None
        long_29 = relative_coords_29.long()
        relative_coords_29 = None
        relative_position_width_14 = rel_pos_resized_59[long_29]
        rel_pos_resized_59 = long_29 = None
        reshaped_query_14 = query_28.reshape(400, 14, 14, 80)
        rel_h_14 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_14, relative_position_height_14
        )
        relative_position_height_14 = None
        rel_w_14 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_14, relative_position_width_14
        )
        reshaped_query_14 = relative_position_width_14 = None
        getitem_184 = rel_h_14[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_14 = None
        getitem_185 = rel_w_14[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_14 = None
        decomposed_rel_pos_28 = getitem_184 + getitem_185
        getitem_184 = getitem_185 = None
        decomposed_rel_pos_29 = decomposed_rel_pos_28.reshape(25, 16, 196, 196)
        decomposed_rel_pos_28 = None
        query_29 = query_28.view(25, 16, 196, -1)
        query_28 = None
        key_29 = key_28.view(25, 16, 196, -1)
        key_28 = None
        value_29 = value_28.view(25, 16, 196, -1)
        value_28 = None
        attn_output_42 = torch._C._nn.scaled_dot_product_attention(
            query_29, key_29, value_29, attn_mask=decomposed_rel_pos_29
        )
        query_29 = key_29 = value_29 = decomposed_rel_pos_29 = None
        view_59 = attn_output_42.view(25, 16, 14, 14, -1)
        attn_output_42 = None
        permute_118 = view_59.permute(0, 2, 3, 1, 4)
        view_59 = None
        attn_output_43 = permute_118.reshape(25, 14, 14, -1)
        permute_118 = None
        attn_output_44 = torch._C._nn.linear(
            attn_output_43,
            l_self_modules_vision_encoder_modules_layers_modules_14_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_14_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_43 = l_self_modules_vision_encoder_modules_layers_modules_14_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_14_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_153 = attn_output_44.reshape(1, 5, 5, 14, 14, -1)
        attn_output_44 = None
        permute_119 = hidden_states_153.permute(0, 1, 3, 2, 4, 5)
        hidden_states_153 = None
        contiguous_40 = permute_119.contiguous()
        permute_119 = None
        hidden_states_154 = contiguous_40.reshape(1, 70, 70, -1)
        contiguous_40 = None
        getitem_186 = hidden_states_154[
            (
                slice(None, None, None),
                slice(None, 64, None),
                slice(None, 64, None),
                slice(None, None, None),
            )
        ]
        hidden_states_154 = None
        hidden_states_155 = getitem_186.contiguous()
        getitem_186 = None
        hidden_states_156 = hidden_states_149 + hidden_states_155
        hidden_states_149 = hidden_states_155 = None
        item_30 = (
            l_self_modules_vision_encoder_modules_layers_modules_14_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_14_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_14 = torch.nn.functional.layer_norm(
            hidden_states_156,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_bias_,
            item_30,
        )
        l_self_modules_vision_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_bias_ = (item_30) = (
            None
        )
        hidden_states_157 = torch._C._nn.linear(
            layernorm_output_14,
            l_self_modules_vision_encoder_modules_layers_modules_14_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_14_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_14 = l_self_modules_vision_encoder_modules_layers_modules_14_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_14_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_158 = torch._C._nn.gelu(hidden_states_157)
        hidden_states_157 = None
        hidden_states_159 = torch._C._nn.linear(
            hidden_states_158,
            l_self_modules_vision_encoder_modules_layers_modules_14_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_14_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_158 = l_self_modules_vision_encoder_modules_layers_modules_14_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_14_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_160 = hidden_states_156 + hidden_states_159
        hidden_states_156 = hidden_states_159 = None
        item_31 = (
            l_self_modules_vision_encoder_modules_layers_modules_15_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_15_modules_layer_norm1_eps = (
            None
        )
        hidden_states_161 = torch.nn.functional.layer_norm(
            hidden_states_160,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_bias_,
            item_31,
        )
        l_self_modules_vision_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_bias_ = (item_31) = (
            None
        )
        linear_60 = torch._C._nn.linear(
            hidden_states_161,
            l_self_modules_vision_encoder_modules_layers_modules_15_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_15_modules_attn_modules_qkv_parameters_bias_,
        )
        hidden_states_161 = l_self_modules_vision_encoder_modules_layers_modules_15_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_15_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_191 = linear_60.reshape(1, 4096, 3, 16, -1)
        linear_60 = None
        qkv_15 = reshape_191.permute(2, 0, 3, 1, 4)
        reshape_191 = None
        reshape_192 = qkv_15.reshape(3, 16, 4096, -1)
        qkv_15 = None
        unbind_15 = reshape_192.unbind(0)
        reshape_192 = None
        query_30 = unbind_15[0]
        key_30 = unbind_15[1]
        value_30 = unbind_15[2]
        unbind_15 = None
        reshape_193 = l_self_modules_vision_encoder_modules_layers_modules_15_modules_attn_parameters_rel_pos_h_.reshape(
            1, 127, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_15_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_121 = reshape_193.permute(0, 2, 1)
        reshape_193 = None
        rel_pos_resized_60 = torch.nn.functional.interpolate(
            permute_121, size=127, mode="linear"
        )
        permute_121 = None
        reshape_194 = rel_pos_resized_60.reshape(-1, 127)
        rel_pos_resized_60 = None
        rel_pos_resized_61 = reshape_194.permute(1, 0)
        reshape_194 = None
        arange_60 = torch.arange(64)
        getitem_190 = arange_60[(slice(None, None, None), None)]
        arange_60 = None
        q_coords_30 = getitem_190 * 1.0
        getitem_190 = None
        arange_61 = torch.arange(64)
        getitem_191 = arange_61[(None, slice(None, None, None))]
        arange_61 = None
        k_coords_30 = getitem_191 * 1.0
        getitem_191 = None
        sub_33 = q_coords_30 - k_coords_30
        q_coords_30 = k_coords_30 = None
        relative_coords_30 = sub_33 + 63.0
        sub_33 = None
        long_30 = relative_coords_30.long()
        relative_coords_30 = None
        relative_position_height_15 = rel_pos_resized_61[long_30]
        rel_pos_resized_61 = long_30 = None
        reshape_195 = l_self_modules_vision_encoder_modules_layers_modules_15_modules_attn_parameters_rel_pos_w_.reshape(
            1, 127, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_15_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_123 = reshape_195.permute(0, 2, 1)
        reshape_195 = None
        rel_pos_resized_62 = torch.nn.functional.interpolate(
            permute_123, size=127, mode="linear"
        )
        permute_123 = None
        reshape_196 = rel_pos_resized_62.reshape(-1, 127)
        rel_pos_resized_62 = None
        rel_pos_resized_63 = reshape_196.permute(1, 0)
        reshape_196 = None
        arange_62 = torch.arange(64)
        getitem_193 = arange_62[(slice(None, None, None), None)]
        arange_62 = None
        q_coords_31 = getitem_193 * 1.0
        getitem_193 = None
        arange_63 = torch.arange(64)
        getitem_194 = arange_63[(None, slice(None, None, None))]
        arange_63 = None
        k_coords_31 = getitem_194 * 1.0
        getitem_194 = None
        sub_34 = q_coords_31 - k_coords_31
        q_coords_31 = k_coords_31 = None
        relative_coords_31 = sub_34 + 63.0
        sub_34 = None
        long_31 = relative_coords_31.long()
        relative_coords_31 = None
        relative_position_width_15 = rel_pos_resized_63[long_31]
        rel_pos_resized_63 = long_31 = None
        reshaped_query_15 = query_30.reshape(16, 64, 64, 80)
        rel_h_15 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_15, relative_position_height_15
        )
        relative_position_height_15 = None
        rel_w_15 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_15, relative_position_width_15
        )
        reshaped_query_15 = relative_position_width_15 = None
        getitem_196 = rel_h_15[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_15 = None
        getitem_197 = rel_w_15[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_15 = None
        decomposed_rel_pos_30 = getitem_196 + getitem_197
        getitem_196 = getitem_197 = None
        decomposed_rel_pos_31 = decomposed_rel_pos_30.reshape(1, 16, 4096, 4096)
        decomposed_rel_pos_30 = None
        query_31 = query_30.view(1, 16, 4096, -1)
        query_30 = None
        key_31 = key_30.view(1, 16, 4096, -1)
        key_30 = None
        value_31 = value_30.view(1, 16, 4096, -1)
        value_30 = None
        attn_output_45 = torch._C._nn.scaled_dot_product_attention(
            query_31, key_31, value_31, attn_mask=decomposed_rel_pos_31
        )
        query_31 = key_31 = value_31 = decomposed_rel_pos_31 = None
        view_63 = attn_output_45.view(1, 16, 64, 64, -1)
        attn_output_45 = None
        permute_125 = view_63.permute(0, 2, 3, 1, 4)
        view_63 = None
        attn_output_46 = permute_125.reshape(1, 64, 64, -1)
        permute_125 = None
        attn_output_47 = torch._C._nn.linear(
            attn_output_46,
            l_self_modules_vision_encoder_modules_layers_modules_15_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_15_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_46 = l_self_modules_vision_encoder_modules_layers_modules_15_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_15_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_162 = hidden_states_160 + attn_output_47
        hidden_states_160 = attn_output_47 = None
        item_32 = (
            l_self_modules_vision_encoder_modules_layers_modules_15_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_15_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_15 = torch.nn.functional.layer_norm(
            hidden_states_162,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_bias_,
            item_32,
        )
        l_self_modules_vision_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_bias_ = (item_32) = (
            None
        )
        hidden_states_163 = torch._C._nn.linear(
            layernorm_output_15,
            l_self_modules_vision_encoder_modules_layers_modules_15_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_15_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_15 = l_self_modules_vision_encoder_modules_layers_modules_15_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_15_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_164 = torch._C._nn.gelu(hidden_states_163)
        hidden_states_163 = None
        hidden_states_165 = torch._C._nn.linear(
            hidden_states_164,
            l_self_modules_vision_encoder_modules_layers_modules_15_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_15_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_164 = l_self_modules_vision_encoder_modules_layers_modules_15_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_15_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_166 = hidden_states_162 + hidden_states_165
        hidden_states_162 = hidden_states_165 = None
        item_33 = (
            l_self_modules_vision_encoder_modules_layers_modules_16_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_16_modules_layer_norm1_eps = (
            None
        )
        hidden_states_167 = torch.nn.functional.layer_norm(
            hidden_states_166,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_bias_,
            item_33,
        )
        l_self_modules_vision_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_bias_ = (item_33) = (
            None
        )
        hidden_states_168 = torch._C._nn.pad(
            hidden_states_167, (0, 0, 0, 6, 0, 6), "constant", None
        )
        hidden_states_167 = None
        hidden_states_169 = hidden_states_168.reshape(1, 5, 14, 5, 14, 1280)
        hidden_states_168 = None
        permute_126 = hidden_states_169.permute(0, 1, 3, 2, 4, 5)
        hidden_states_169 = None
        contiguous_42 = permute_126.contiguous()
        permute_126 = None
        windows_14 = contiguous_42.reshape(-1, 14, 14, 1280)
        contiguous_42 = None
        linear_64 = torch._C._nn.linear(
            windows_14,
            l_self_modules_vision_encoder_modules_layers_modules_16_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_16_modules_attn_modules_qkv_parameters_bias_,
        )
        windows_14 = l_self_modules_vision_encoder_modules_layers_modules_16_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_16_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_202 = linear_64.reshape(25, 196, 3, 16, -1)
        linear_64 = None
        qkv_16 = reshape_202.permute(2, 0, 3, 1, 4)
        reshape_202 = None
        reshape_203 = qkv_16.reshape(3, 400, 196, -1)
        qkv_16 = None
        unbind_16 = reshape_203.unbind(0)
        reshape_203 = None
        query_32 = unbind_16[0]
        key_32 = unbind_16[1]
        value_32 = unbind_16[2]
        unbind_16 = None
        reshape_204 = l_self_modules_vision_encoder_modules_layers_modules_16_modules_attn_parameters_rel_pos_h_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_16_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_128 = reshape_204.permute(0, 2, 1)
        reshape_204 = None
        rel_pos_resized_64 = torch.nn.functional.interpolate(
            permute_128, size=27, mode="linear"
        )
        permute_128 = None
        reshape_205 = rel_pos_resized_64.reshape(-1, 27)
        rel_pos_resized_64 = None
        rel_pos_resized_65 = reshape_205.permute(1, 0)
        reshape_205 = None
        arange_64 = torch.arange(14)
        getitem_201 = arange_64[(slice(None, None, None), None)]
        arange_64 = None
        q_coords_32 = getitem_201 * 1.0
        getitem_201 = None
        arange_65 = torch.arange(14)
        getitem_202 = arange_65[(None, slice(None, None, None))]
        arange_65 = None
        k_coords_32 = getitem_202 * 1.0
        getitem_202 = None
        sub_35 = q_coords_32 - k_coords_32
        q_coords_32 = k_coords_32 = None
        relative_coords_32 = sub_35 + 13.0
        sub_35 = None
        long_32 = relative_coords_32.long()
        relative_coords_32 = None
        relative_position_height_16 = rel_pos_resized_65[long_32]
        rel_pos_resized_65 = long_32 = None
        reshape_206 = l_self_modules_vision_encoder_modules_layers_modules_16_modules_attn_parameters_rel_pos_w_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_16_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_130 = reshape_206.permute(0, 2, 1)
        reshape_206 = None
        rel_pos_resized_66 = torch.nn.functional.interpolate(
            permute_130, size=27, mode="linear"
        )
        permute_130 = None
        reshape_207 = rel_pos_resized_66.reshape(-1, 27)
        rel_pos_resized_66 = None
        rel_pos_resized_67 = reshape_207.permute(1, 0)
        reshape_207 = None
        arange_66 = torch.arange(14)
        getitem_204 = arange_66[(slice(None, None, None), None)]
        arange_66 = None
        q_coords_33 = getitem_204 * 1.0
        getitem_204 = None
        arange_67 = torch.arange(14)
        getitem_205 = arange_67[(None, slice(None, None, None))]
        arange_67 = None
        k_coords_33 = getitem_205 * 1.0
        getitem_205 = None
        sub_36 = q_coords_33 - k_coords_33
        q_coords_33 = k_coords_33 = None
        relative_coords_33 = sub_36 + 13.0
        sub_36 = None
        long_33 = relative_coords_33.long()
        relative_coords_33 = None
        relative_position_width_16 = rel_pos_resized_67[long_33]
        rel_pos_resized_67 = long_33 = None
        reshaped_query_16 = query_32.reshape(400, 14, 14, 80)
        rel_h_16 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_16, relative_position_height_16
        )
        relative_position_height_16 = None
        rel_w_16 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_16, relative_position_width_16
        )
        reshaped_query_16 = relative_position_width_16 = None
        getitem_207 = rel_h_16[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_16 = None
        getitem_208 = rel_w_16[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_16 = None
        decomposed_rel_pos_32 = getitem_207 + getitem_208
        getitem_207 = getitem_208 = None
        decomposed_rel_pos_33 = decomposed_rel_pos_32.reshape(25, 16, 196, 196)
        decomposed_rel_pos_32 = None
        query_33 = query_32.view(25, 16, 196, -1)
        query_32 = None
        key_33 = key_32.view(25, 16, 196, -1)
        key_32 = None
        value_33 = value_32.view(25, 16, 196, -1)
        value_32 = None
        attn_output_48 = torch._C._nn.scaled_dot_product_attention(
            query_33, key_33, value_33, attn_mask=decomposed_rel_pos_33
        )
        query_33 = key_33 = value_33 = decomposed_rel_pos_33 = None
        view_67 = attn_output_48.view(25, 16, 14, 14, -1)
        attn_output_48 = None
        permute_132 = view_67.permute(0, 2, 3, 1, 4)
        view_67 = None
        attn_output_49 = permute_132.reshape(25, 14, 14, -1)
        permute_132 = None
        attn_output_50 = torch._C._nn.linear(
            attn_output_49,
            l_self_modules_vision_encoder_modules_layers_modules_16_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_16_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_49 = l_self_modules_vision_encoder_modules_layers_modules_16_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_16_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_170 = attn_output_50.reshape(1, 5, 5, 14, 14, -1)
        attn_output_50 = None
        permute_133 = hidden_states_170.permute(0, 1, 3, 2, 4, 5)
        hidden_states_170 = None
        contiguous_43 = permute_133.contiguous()
        permute_133 = None
        hidden_states_171 = contiguous_43.reshape(1, 70, 70, -1)
        contiguous_43 = None
        getitem_209 = hidden_states_171[
            (
                slice(None, None, None),
                slice(None, 64, None),
                slice(None, 64, None),
                slice(None, None, None),
            )
        ]
        hidden_states_171 = None
        hidden_states_172 = getitem_209.contiguous()
        getitem_209 = None
        hidden_states_173 = hidden_states_166 + hidden_states_172
        hidden_states_166 = hidden_states_172 = None
        item_34 = (
            l_self_modules_vision_encoder_modules_layers_modules_16_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_16_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_16 = torch.nn.functional.layer_norm(
            hidden_states_173,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_bias_,
            item_34,
        )
        l_self_modules_vision_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_bias_ = (item_34) = (
            None
        )
        hidden_states_174 = torch._C._nn.linear(
            layernorm_output_16,
            l_self_modules_vision_encoder_modules_layers_modules_16_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_16_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_16 = l_self_modules_vision_encoder_modules_layers_modules_16_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_16_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_175 = torch._C._nn.gelu(hidden_states_174)
        hidden_states_174 = None
        hidden_states_176 = torch._C._nn.linear(
            hidden_states_175,
            l_self_modules_vision_encoder_modules_layers_modules_16_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_16_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_175 = l_self_modules_vision_encoder_modules_layers_modules_16_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_16_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_177 = hidden_states_173 + hidden_states_176
        hidden_states_173 = hidden_states_176 = None
        item_35 = (
            l_self_modules_vision_encoder_modules_layers_modules_17_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_17_modules_layer_norm1_eps = (
            None
        )
        hidden_states_178 = torch.nn.functional.layer_norm(
            hidden_states_177,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_bias_,
            item_35,
        )
        l_self_modules_vision_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_bias_ = (item_35) = (
            None
        )
        hidden_states_179 = torch._C._nn.pad(
            hidden_states_178, (0, 0, 0, 6, 0, 6), "constant", None
        )
        hidden_states_178 = None
        hidden_states_180 = hidden_states_179.reshape(1, 5, 14, 5, 14, 1280)
        hidden_states_179 = None
        permute_134 = hidden_states_180.permute(0, 1, 3, 2, 4, 5)
        hidden_states_180 = None
        contiguous_45 = permute_134.contiguous()
        permute_134 = None
        windows_15 = contiguous_45.reshape(-1, 14, 14, 1280)
        contiguous_45 = None
        linear_68 = torch._C._nn.linear(
            windows_15,
            l_self_modules_vision_encoder_modules_layers_modules_17_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_17_modules_attn_modules_qkv_parameters_bias_,
        )
        windows_15 = l_self_modules_vision_encoder_modules_layers_modules_17_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_17_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_215 = linear_68.reshape(25, 196, 3, 16, -1)
        linear_68 = None
        qkv_17 = reshape_215.permute(2, 0, 3, 1, 4)
        reshape_215 = None
        reshape_216 = qkv_17.reshape(3, 400, 196, -1)
        qkv_17 = None
        unbind_17 = reshape_216.unbind(0)
        reshape_216 = None
        query_34 = unbind_17[0]
        key_34 = unbind_17[1]
        value_34 = unbind_17[2]
        unbind_17 = None
        reshape_217 = l_self_modules_vision_encoder_modules_layers_modules_17_modules_attn_parameters_rel_pos_h_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_17_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_136 = reshape_217.permute(0, 2, 1)
        reshape_217 = None
        rel_pos_resized_68 = torch.nn.functional.interpolate(
            permute_136, size=27, mode="linear"
        )
        permute_136 = None
        reshape_218 = rel_pos_resized_68.reshape(-1, 27)
        rel_pos_resized_68 = None
        rel_pos_resized_69 = reshape_218.permute(1, 0)
        reshape_218 = None
        arange_68 = torch.arange(14)
        getitem_213 = arange_68[(slice(None, None, None), None)]
        arange_68 = None
        q_coords_34 = getitem_213 * 1.0
        getitem_213 = None
        arange_69 = torch.arange(14)
        getitem_214 = arange_69[(None, slice(None, None, None))]
        arange_69 = None
        k_coords_34 = getitem_214 * 1.0
        getitem_214 = None
        sub_37 = q_coords_34 - k_coords_34
        q_coords_34 = k_coords_34 = None
        relative_coords_34 = sub_37 + 13.0
        sub_37 = None
        long_34 = relative_coords_34.long()
        relative_coords_34 = None
        relative_position_height_17 = rel_pos_resized_69[long_34]
        rel_pos_resized_69 = long_34 = None
        reshape_219 = l_self_modules_vision_encoder_modules_layers_modules_17_modules_attn_parameters_rel_pos_w_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_17_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_138 = reshape_219.permute(0, 2, 1)
        reshape_219 = None
        rel_pos_resized_70 = torch.nn.functional.interpolate(
            permute_138, size=27, mode="linear"
        )
        permute_138 = None
        reshape_220 = rel_pos_resized_70.reshape(-1, 27)
        rel_pos_resized_70 = None
        rel_pos_resized_71 = reshape_220.permute(1, 0)
        reshape_220 = None
        arange_70 = torch.arange(14)
        getitem_216 = arange_70[(slice(None, None, None), None)]
        arange_70 = None
        q_coords_35 = getitem_216 * 1.0
        getitem_216 = None
        arange_71 = torch.arange(14)
        getitem_217 = arange_71[(None, slice(None, None, None))]
        arange_71 = None
        k_coords_35 = getitem_217 * 1.0
        getitem_217 = None
        sub_38 = q_coords_35 - k_coords_35
        q_coords_35 = k_coords_35 = None
        relative_coords_35 = sub_38 + 13.0
        sub_38 = None
        long_35 = relative_coords_35.long()
        relative_coords_35 = None
        relative_position_width_17 = rel_pos_resized_71[long_35]
        rel_pos_resized_71 = long_35 = None
        reshaped_query_17 = query_34.reshape(400, 14, 14, 80)
        rel_h_17 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_17, relative_position_height_17
        )
        relative_position_height_17 = None
        rel_w_17 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_17, relative_position_width_17
        )
        reshaped_query_17 = relative_position_width_17 = None
        getitem_219 = rel_h_17[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_17 = None
        getitem_220 = rel_w_17[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_17 = None
        decomposed_rel_pos_34 = getitem_219 + getitem_220
        getitem_219 = getitem_220 = None
        decomposed_rel_pos_35 = decomposed_rel_pos_34.reshape(25, 16, 196, 196)
        decomposed_rel_pos_34 = None
        query_35 = query_34.view(25, 16, 196, -1)
        query_34 = None
        key_35 = key_34.view(25, 16, 196, -1)
        key_34 = None
        value_35 = value_34.view(25, 16, 196, -1)
        value_34 = None
        attn_output_51 = torch._C._nn.scaled_dot_product_attention(
            query_35, key_35, value_35, attn_mask=decomposed_rel_pos_35
        )
        query_35 = key_35 = value_35 = decomposed_rel_pos_35 = None
        view_71 = attn_output_51.view(25, 16, 14, 14, -1)
        attn_output_51 = None
        permute_140 = view_71.permute(0, 2, 3, 1, 4)
        view_71 = None
        attn_output_52 = permute_140.reshape(25, 14, 14, -1)
        permute_140 = None
        attn_output_53 = torch._C._nn.linear(
            attn_output_52,
            l_self_modules_vision_encoder_modules_layers_modules_17_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_17_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_52 = l_self_modules_vision_encoder_modules_layers_modules_17_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_17_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_181 = attn_output_53.reshape(1, 5, 5, 14, 14, -1)
        attn_output_53 = None
        permute_141 = hidden_states_181.permute(0, 1, 3, 2, 4, 5)
        hidden_states_181 = None
        contiguous_46 = permute_141.contiguous()
        permute_141 = None
        hidden_states_182 = contiguous_46.reshape(1, 70, 70, -1)
        contiguous_46 = None
        getitem_221 = hidden_states_182[
            (
                slice(None, None, None),
                slice(None, 64, None),
                slice(None, 64, None),
                slice(None, None, None),
            )
        ]
        hidden_states_182 = None
        hidden_states_183 = getitem_221.contiguous()
        getitem_221 = None
        hidden_states_184 = hidden_states_177 + hidden_states_183
        hidden_states_177 = hidden_states_183 = None
        item_36 = (
            l_self_modules_vision_encoder_modules_layers_modules_17_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_17_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_17 = torch.nn.functional.layer_norm(
            hidden_states_184,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_bias_,
            item_36,
        )
        l_self_modules_vision_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_bias_ = (item_36) = (
            None
        )
        hidden_states_185 = torch._C._nn.linear(
            layernorm_output_17,
            l_self_modules_vision_encoder_modules_layers_modules_17_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_17_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_17 = l_self_modules_vision_encoder_modules_layers_modules_17_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_17_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_186 = torch._C._nn.gelu(hidden_states_185)
        hidden_states_185 = None
        hidden_states_187 = torch._C._nn.linear(
            hidden_states_186,
            l_self_modules_vision_encoder_modules_layers_modules_17_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_17_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_186 = l_self_modules_vision_encoder_modules_layers_modules_17_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_17_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_188 = hidden_states_184 + hidden_states_187
        hidden_states_184 = hidden_states_187 = None
        item_37 = (
            l_self_modules_vision_encoder_modules_layers_modules_18_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_18_modules_layer_norm1_eps = (
            None
        )
        hidden_states_189 = torch.nn.functional.layer_norm(
            hidden_states_188,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_bias_,
            item_37,
        )
        l_self_modules_vision_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_bias_ = (item_37) = (
            None
        )
        hidden_states_190 = torch._C._nn.pad(
            hidden_states_189, (0, 0, 0, 6, 0, 6), "constant", None
        )
        hidden_states_189 = None
        hidden_states_191 = hidden_states_190.reshape(1, 5, 14, 5, 14, 1280)
        hidden_states_190 = None
        permute_142 = hidden_states_191.permute(0, 1, 3, 2, 4, 5)
        hidden_states_191 = None
        contiguous_48 = permute_142.contiguous()
        permute_142 = None
        windows_16 = contiguous_48.reshape(-1, 14, 14, 1280)
        contiguous_48 = None
        linear_72 = torch._C._nn.linear(
            windows_16,
            l_self_modules_vision_encoder_modules_layers_modules_18_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_18_modules_attn_modules_qkv_parameters_bias_,
        )
        windows_16 = l_self_modules_vision_encoder_modules_layers_modules_18_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_18_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_228 = linear_72.reshape(25, 196, 3, 16, -1)
        linear_72 = None
        qkv_18 = reshape_228.permute(2, 0, 3, 1, 4)
        reshape_228 = None
        reshape_229 = qkv_18.reshape(3, 400, 196, -1)
        qkv_18 = None
        unbind_18 = reshape_229.unbind(0)
        reshape_229 = None
        query_36 = unbind_18[0]
        key_36 = unbind_18[1]
        value_36 = unbind_18[2]
        unbind_18 = None
        reshape_230 = l_self_modules_vision_encoder_modules_layers_modules_18_modules_attn_parameters_rel_pos_h_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_18_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_144 = reshape_230.permute(0, 2, 1)
        reshape_230 = None
        rel_pos_resized_72 = torch.nn.functional.interpolate(
            permute_144, size=27, mode="linear"
        )
        permute_144 = None
        reshape_231 = rel_pos_resized_72.reshape(-1, 27)
        rel_pos_resized_72 = None
        rel_pos_resized_73 = reshape_231.permute(1, 0)
        reshape_231 = None
        arange_72 = torch.arange(14)
        getitem_225 = arange_72[(slice(None, None, None), None)]
        arange_72 = None
        q_coords_36 = getitem_225 * 1.0
        getitem_225 = None
        arange_73 = torch.arange(14)
        getitem_226 = arange_73[(None, slice(None, None, None))]
        arange_73 = None
        k_coords_36 = getitem_226 * 1.0
        getitem_226 = None
        sub_39 = q_coords_36 - k_coords_36
        q_coords_36 = k_coords_36 = None
        relative_coords_36 = sub_39 + 13.0
        sub_39 = None
        long_36 = relative_coords_36.long()
        relative_coords_36 = None
        relative_position_height_18 = rel_pos_resized_73[long_36]
        rel_pos_resized_73 = long_36 = None
        reshape_232 = l_self_modules_vision_encoder_modules_layers_modules_18_modules_attn_parameters_rel_pos_w_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_18_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_146 = reshape_232.permute(0, 2, 1)
        reshape_232 = None
        rel_pos_resized_74 = torch.nn.functional.interpolate(
            permute_146, size=27, mode="linear"
        )
        permute_146 = None
        reshape_233 = rel_pos_resized_74.reshape(-1, 27)
        rel_pos_resized_74 = None
        rel_pos_resized_75 = reshape_233.permute(1, 0)
        reshape_233 = None
        arange_74 = torch.arange(14)
        getitem_228 = arange_74[(slice(None, None, None), None)]
        arange_74 = None
        q_coords_37 = getitem_228 * 1.0
        getitem_228 = None
        arange_75 = torch.arange(14)
        getitem_229 = arange_75[(None, slice(None, None, None))]
        arange_75 = None
        k_coords_37 = getitem_229 * 1.0
        getitem_229 = None
        sub_40 = q_coords_37 - k_coords_37
        q_coords_37 = k_coords_37 = None
        relative_coords_37 = sub_40 + 13.0
        sub_40 = None
        long_37 = relative_coords_37.long()
        relative_coords_37 = None
        relative_position_width_18 = rel_pos_resized_75[long_37]
        rel_pos_resized_75 = long_37 = None
        reshaped_query_18 = query_36.reshape(400, 14, 14, 80)
        rel_h_18 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_18, relative_position_height_18
        )
        relative_position_height_18 = None
        rel_w_18 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_18, relative_position_width_18
        )
        reshaped_query_18 = relative_position_width_18 = None
        getitem_231 = rel_h_18[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_18 = None
        getitem_232 = rel_w_18[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_18 = None
        decomposed_rel_pos_36 = getitem_231 + getitem_232
        getitem_231 = getitem_232 = None
        decomposed_rel_pos_37 = decomposed_rel_pos_36.reshape(25, 16, 196, 196)
        decomposed_rel_pos_36 = None
        query_37 = query_36.view(25, 16, 196, -1)
        query_36 = None
        key_37 = key_36.view(25, 16, 196, -1)
        key_36 = None
        value_37 = value_36.view(25, 16, 196, -1)
        value_36 = None
        attn_output_54 = torch._C._nn.scaled_dot_product_attention(
            query_37, key_37, value_37, attn_mask=decomposed_rel_pos_37
        )
        query_37 = key_37 = value_37 = decomposed_rel_pos_37 = None
        view_75 = attn_output_54.view(25, 16, 14, 14, -1)
        attn_output_54 = None
        permute_148 = view_75.permute(0, 2, 3, 1, 4)
        view_75 = None
        attn_output_55 = permute_148.reshape(25, 14, 14, -1)
        permute_148 = None
        attn_output_56 = torch._C._nn.linear(
            attn_output_55,
            l_self_modules_vision_encoder_modules_layers_modules_18_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_18_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_55 = l_self_modules_vision_encoder_modules_layers_modules_18_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_18_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_192 = attn_output_56.reshape(1, 5, 5, 14, 14, -1)
        attn_output_56 = None
        permute_149 = hidden_states_192.permute(0, 1, 3, 2, 4, 5)
        hidden_states_192 = None
        contiguous_49 = permute_149.contiguous()
        permute_149 = None
        hidden_states_193 = contiguous_49.reshape(1, 70, 70, -1)
        contiguous_49 = None
        getitem_233 = hidden_states_193[
            (
                slice(None, None, None),
                slice(None, 64, None),
                slice(None, 64, None),
                slice(None, None, None),
            )
        ]
        hidden_states_193 = None
        hidden_states_194 = getitem_233.contiguous()
        getitem_233 = None
        hidden_states_195 = hidden_states_188 + hidden_states_194
        hidden_states_188 = hidden_states_194 = None
        item_38 = (
            l_self_modules_vision_encoder_modules_layers_modules_18_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_18_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_18 = torch.nn.functional.layer_norm(
            hidden_states_195,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_bias_,
            item_38,
        )
        l_self_modules_vision_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_bias_ = (item_38) = (
            None
        )
        hidden_states_196 = torch._C._nn.linear(
            layernorm_output_18,
            l_self_modules_vision_encoder_modules_layers_modules_18_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_18_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_18 = l_self_modules_vision_encoder_modules_layers_modules_18_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_18_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_197 = torch._C._nn.gelu(hidden_states_196)
        hidden_states_196 = None
        hidden_states_198 = torch._C._nn.linear(
            hidden_states_197,
            l_self_modules_vision_encoder_modules_layers_modules_18_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_18_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_197 = l_self_modules_vision_encoder_modules_layers_modules_18_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_18_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_199 = hidden_states_195 + hidden_states_198
        hidden_states_195 = hidden_states_198 = None
        item_39 = (
            l_self_modules_vision_encoder_modules_layers_modules_19_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_19_modules_layer_norm1_eps = (
            None
        )
        hidden_states_200 = torch.nn.functional.layer_norm(
            hidden_states_199,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_bias_,
            item_39,
        )
        l_self_modules_vision_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_bias_ = (item_39) = (
            None
        )
        hidden_states_201 = torch._C._nn.pad(
            hidden_states_200, (0, 0, 0, 6, 0, 6), "constant", None
        )
        hidden_states_200 = None
        hidden_states_202 = hidden_states_201.reshape(1, 5, 14, 5, 14, 1280)
        hidden_states_201 = None
        permute_150 = hidden_states_202.permute(0, 1, 3, 2, 4, 5)
        hidden_states_202 = None
        contiguous_51 = permute_150.contiguous()
        permute_150 = None
        windows_17 = contiguous_51.reshape(-1, 14, 14, 1280)
        contiguous_51 = None
        linear_76 = torch._C._nn.linear(
            windows_17,
            l_self_modules_vision_encoder_modules_layers_modules_19_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_19_modules_attn_modules_qkv_parameters_bias_,
        )
        windows_17 = l_self_modules_vision_encoder_modules_layers_modules_19_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_19_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_241 = linear_76.reshape(25, 196, 3, 16, -1)
        linear_76 = None
        qkv_19 = reshape_241.permute(2, 0, 3, 1, 4)
        reshape_241 = None
        reshape_242 = qkv_19.reshape(3, 400, 196, -1)
        qkv_19 = None
        unbind_19 = reshape_242.unbind(0)
        reshape_242 = None
        query_38 = unbind_19[0]
        key_38 = unbind_19[1]
        value_38 = unbind_19[2]
        unbind_19 = None
        reshape_243 = l_self_modules_vision_encoder_modules_layers_modules_19_modules_attn_parameters_rel_pos_h_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_19_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_152 = reshape_243.permute(0, 2, 1)
        reshape_243 = None
        rel_pos_resized_76 = torch.nn.functional.interpolate(
            permute_152, size=27, mode="linear"
        )
        permute_152 = None
        reshape_244 = rel_pos_resized_76.reshape(-1, 27)
        rel_pos_resized_76 = None
        rel_pos_resized_77 = reshape_244.permute(1, 0)
        reshape_244 = None
        arange_76 = torch.arange(14)
        getitem_237 = arange_76[(slice(None, None, None), None)]
        arange_76 = None
        q_coords_38 = getitem_237 * 1.0
        getitem_237 = None
        arange_77 = torch.arange(14)
        getitem_238 = arange_77[(None, slice(None, None, None))]
        arange_77 = None
        k_coords_38 = getitem_238 * 1.0
        getitem_238 = None
        sub_41 = q_coords_38 - k_coords_38
        q_coords_38 = k_coords_38 = None
        relative_coords_38 = sub_41 + 13.0
        sub_41 = None
        long_38 = relative_coords_38.long()
        relative_coords_38 = None
        relative_position_height_19 = rel_pos_resized_77[long_38]
        rel_pos_resized_77 = long_38 = None
        reshape_245 = l_self_modules_vision_encoder_modules_layers_modules_19_modules_attn_parameters_rel_pos_w_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_19_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_154 = reshape_245.permute(0, 2, 1)
        reshape_245 = None
        rel_pos_resized_78 = torch.nn.functional.interpolate(
            permute_154, size=27, mode="linear"
        )
        permute_154 = None
        reshape_246 = rel_pos_resized_78.reshape(-1, 27)
        rel_pos_resized_78 = None
        rel_pos_resized_79 = reshape_246.permute(1, 0)
        reshape_246 = None
        arange_78 = torch.arange(14)
        getitem_240 = arange_78[(slice(None, None, None), None)]
        arange_78 = None
        q_coords_39 = getitem_240 * 1.0
        getitem_240 = None
        arange_79 = torch.arange(14)
        getitem_241 = arange_79[(None, slice(None, None, None))]
        arange_79 = None
        k_coords_39 = getitem_241 * 1.0
        getitem_241 = None
        sub_42 = q_coords_39 - k_coords_39
        q_coords_39 = k_coords_39 = None
        relative_coords_39 = sub_42 + 13.0
        sub_42 = None
        long_39 = relative_coords_39.long()
        relative_coords_39 = None
        relative_position_width_19 = rel_pos_resized_79[long_39]
        rel_pos_resized_79 = long_39 = None
        reshaped_query_19 = query_38.reshape(400, 14, 14, 80)
        rel_h_19 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_19, relative_position_height_19
        )
        relative_position_height_19 = None
        rel_w_19 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_19, relative_position_width_19
        )
        reshaped_query_19 = relative_position_width_19 = None
        getitem_243 = rel_h_19[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_19 = None
        getitem_244 = rel_w_19[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_19 = None
        decomposed_rel_pos_38 = getitem_243 + getitem_244
        getitem_243 = getitem_244 = None
        decomposed_rel_pos_39 = decomposed_rel_pos_38.reshape(25, 16, 196, 196)
        decomposed_rel_pos_38 = None
        query_39 = query_38.view(25, 16, 196, -1)
        query_38 = None
        key_39 = key_38.view(25, 16, 196, -1)
        key_38 = None
        value_39 = value_38.view(25, 16, 196, -1)
        value_38 = None
        attn_output_57 = torch._C._nn.scaled_dot_product_attention(
            query_39, key_39, value_39, attn_mask=decomposed_rel_pos_39
        )
        query_39 = key_39 = value_39 = decomposed_rel_pos_39 = None
        view_79 = attn_output_57.view(25, 16, 14, 14, -1)
        attn_output_57 = None
        permute_156 = view_79.permute(0, 2, 3, 1, 4)
        view_79 = None
        attn_output_58 = permute_156.reshape(25, 14, 14, -1)
        permute_156 = None
        attn_output_59 = torch._C._nn.linear(
            attn_output_58,
            l_self_modules_vision_encoder_modules_layers_modules_19_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_19_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_58 = l_self_modules_vision_encoder_modules_layers_modules_19_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_19_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_203 = attn_output_59.reshape(1, 5, 5, 14, 14, -1)
        attn_output_59 = None
        permute_157 = hidden_states_203.permute(0, 1, 3, 2, 4, 5)
        hidden_states_203 = None
        contiguous_52 = permute_157.contiguous()
        permute_157 = None
        hidden_states_204 = contiguous_52.reshape(1, 70, 70, -1)
        contiguous_52 = None
        getitem_245 = hidden_states_204[
            (
                slice(None, None, None),
                slice(None, 64, None),
                slice(None, 64, None),
                slice(None, None, None),
            )
        ]
        hidden_states_204 = None
        hidden_states_205 = getitem_245.contiguous()
        getitem_245 = None
        hidden_states_206 = hidden_states_199 + hidden_states_205
        hidden_states_199 = hidden_states_205 = None
        item_40 = (
            l_self_modules_vision_encoder_modules_layers_modules_19_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_19_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_19 = torch.nn.functional.layer_norm(
            hidden_states_206,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_bias_,
            item_40,
        )
        l_self_modules_vision_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_bias_ = (item_40) = (
            None
        )
        hidden_states_207 = torch._C._nn.linear(
            layernorm_output_19,
            l_self_modules_vision_encoder_modules_layers_modules_19_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_19_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_19 = l_self_modules_vision_encoder_modules_layers_modules_19_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_19_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_208 = torch._C._nn.gelu(hidden_states_207)
        hidden_states_207 = None
        hidden_states_209 = torch._C._nn.linear(
            hidden_states_208,
            l_self_modules_vision_encoder_modules_layers_modules_19_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_19_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_208 = l_self_modules_vision_encoder_modules_layers_modules_19_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_19_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_210 = hidden_states_206 + hidden_states_209
        hidden_states_206 = hidden_states_209 = None
        item_41 = (
            l_self_modules_vision_encoder_modules_layers_modules_20_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_20_modules_layer_norm1_eps = (
            None
        )
        hidden_states_211 = torch.nn.functional.layer_norm(
            hidden_states_210,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_bias_,
            item_41,
        )
        l_self_modules_vision_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_bias_ = (item_41) = (
            None
        )
        hidden_states_212 = torch._C._nn.pad(
            hidden_states_211, (0, 0, 0, 6, 0, 6), "constant", None
        )
        hidden_states_211 = None
        hidden_states_213 = hidden_states_212.reshape(1, 5, 14, 5, 14, 1280)
        hidden_states_212 = None
        permute_158 = hidden_states_213.permute(0, 1, 3, 2, 4, 5)
        hidden_states_213 = None
        contiguous_54 = permute_158.contiguous()
        permute_158 = None
        windows_18 = contiguous_54.reshape(-1, 14, 14, 1280)
        contiguous_54 = None
        linear_80 = torch._C._nn.linear(
            windows_18,
            l_self_modules_vision_encoder_modules_layers_modules_20_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_20_modules_attn_modules_qkv_parameters_bias_,
        )
        windows_18 = l_self_modules_vision_encoder_modules_layers_modules_20_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_20_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_254 = linear_80.reshape(25, 196, 3, 16, -1)
        linear_80 = None
        qkv_20 = reshape_254.permute(2, 0, 3, 1, 4)
        reshape_254 = None
        reshape_255 = qkv_20.reshape(3, 400, 196, -1)
        qkv_20 = None
        unbind_20 = reshape_255.unbind(0)
        reshape_255 = None
        query_40 = unbind_20[0]
        key_40 = unbind_20[1]
        value_40 = unbind_20[2]
        unbind_20 = None
        reshape_256 = l_self_modules_vision_encoder_modules_layers_modules_20_modules_attn_parameters_rel_pos_h_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_20_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_160 = reshape_256.permute(0, 2, 1)
        reshape_256 = None
        rel_pos_resized_80 = torch.nn.functional.interpolate(
            permute_160, size=27, mode="linear"
        )
        permute_160 = None
        reshape_257 = rel_pos_resized_80.reshape(-1, 27)
        rel_pos_resized_80 = None
        rel_pos_resized_81 = reshape_257.permute(1, 0)
        reshape_257 = None
        arange_80 = torch.arange(14)
        getitem_249 = arange_80[(slice(None, None, None), None)]
        arange_80 = None
        q_coords_40 = getitem_249 * 1.0
        getitem_249 = None
        arange_81 = torch.arange(14)
        getitem_250 = arange_81[(None, slice(None, None, None))]
        arange_81 = None
        k_coords_40 = getitem_250 * 1.0
        getitem_250 = None
        sub_43 = q_coords_40 - k_coords_40
        q_coords_40 = k_coords_40 = None
        relative_coords_40 = sub_43 + 13.0
        sub_43 = None
        long_40 = relative_coords_40.long()
        relative_coords_40 = None
        relative_position_height_20 = rel_pos_resized_81[long_40]
        rel_pos_resized_81 = long_40 = None
        reshape_258 = l_self_modules_vision_encoder_modules_layers_modules_20_modules_attn_parameters_rel_pos_w_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_20_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_162 = reshape_258.permute(0, 2, 1)
        reshape_258 = None
        rel_pos_resized_82 = torch.nn.functional.interpolate(
            permute_162, size=27, mode="linear"
        )
        permute_162 = None
        reshape_259 = rel_pos_resized_82.reshape(-1, 27)
        rel_pos_resized_82 = None
        rel_pos_resized_83 = reshape_259.permute(1, 0)
        reshape_259 = None
        arange_82 = torch.arange(14)
        getitem_252 = arange_82[(slice(None, None, None), None)]
        arange_82 = None
        q_coords_41 = getitem_252 * 1.0
        getitem_252 = None
        arange_83 = torch.arange(14)
        getitem_253 = arange_83[(None, slice(None, None, None))]
        arange_83 = None
        k_coords_41 = getitem_253 * 1.0
        getitem_253 = None
        sub_44 = q_coords_41 - k_coords_41
        q_coords_41 = k_coords_41 = None
        relative_coords_41 = sub_44 + 13.0
        sub_44 = None
        long_41 = relative_coords_41.long()
        relative_coords_41 = None
        relative_position_width_20 = rel_pos_resized_83[long_41]
        rel_pos_resized_83 = long_41 = None
        reshaped_query_20 = query_40.reshape(400, 14, 14, 80)
        rel_h_20 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_20, relative_position_height_20
        )
        relative_position_height_20 = None
        rel_w_20 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_20, relative_position_width_20
        )
        reshaped_query_20 = relative_position_width_20 = None
        getitem_255 = rel_h_20[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_20 = None
        getitem_256 = rel_w_20[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_20 = None
        decomposed_rel_pos_40 = getitem_255 + getitem_256
        getitem_255 = getitem_256 = None
        decomposed_rel_pos_41 = decomposed_rel_pos_40.reshape(25, 16, 196, 196)
        decomposed_rel_pos_40 = None
        query_41 = query_40.view(25, 16, 196, -1)
        query_40 = None
        key_41 = key_40.view(25, 16, 196, -1)
        key_40 = None
        value_41 = value_40.view(25, 16, 196, -1)
        value_40 = None
        attn_output_60 = torch._C._nn.scaled_dot_product_attention(
            query_41, key_41, value_41, attn_mask=decomposed_rel_pos_41
        )
        query_41 = key_41 = value_41 = decomposed_rel_pos_41 = None
        view_83 = attn_output_60.view(25, 16, 14, 14, -1)
        attn_output_60 = None
        permute_164 = view_83.permute(0, 2, 3, 1, 4)
        view_83 = None
        attn_output_61 = permute_164.reshape(25, 14, 14, -1)
        permute_164 = None
        attn_output_62 = torch._C._nn.linear(
            attn_output_61,
            l_self_modules_vision_encoder_modules_layers_modules_20_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_20_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_61 = l_self_modules_vision_encoder_modules_layers_modules_20_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_20_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_214 = attn_output_62.reshape(1, 5, 5, 14, 14, -1)
        attn_output_62 = None
        permute_165 = hidden_states_214.permute(0, 1, 3, 2, 4, 5)
        hidden_states_214 = None
        contiguous_55 = permute_165.contiguous()
        permute_165 = None
        hidden_states_215 = contiguous_55.reshape(1, 70, 70, -1)
        contiguous_55 = None
        getitem_257 = hidden_states_215[
            (
                slice(None, None, None),
                slice(None, 64, None),
                slice(None, 64, None),
                slice(None, None, None),
            )
        ]
        hidden_states_215 = None
        hidden_states_216 = getitem_257.contiguous()
        getitem_257 = None
        hidden_states_217 = hidden_states_210 + hidden_states_216
        hidden_states_210 = hidden_states_216 = None
        item_42 = (
            l_self_modules_vision_encoder_modules_layers_modules_20_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_20_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_20 = torch.nn.functional.layer_norm(
            hidden_states_217,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_bias_,
            item_42,
        )
        l_self_modules_vision_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_bias_ = (item_42) = (
            None
        )
        hidden_states_218 = torch._C._nn.linear(
            layernorm_output_20,
            l_self_modules_vision_encoder_modules_layers_modules_20_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_20_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_20 = l_self_modules_vision_encoder_modules_layers_modules_20_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_20_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_219 = torch._C._nn.gelu(hidden_states_218)
        hidden_states_218 = None
        hidden_states_220 = torch._C._nn.linear(
            hidden_states_219,
            l_self_modules_vision_encoder_modules_layers_modules_20_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_20_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_219 = l_self_modules_vision_encoder_modules_layers_modules_20_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_20_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_221 = hidden_states_217 + hidden_states_220
        hidden_states_217 = hidden_states_220 = None
        item_43 = (
            l_self_modules_vision_encoder_modules_layers_modules_21_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_21_modules_layer_norm1_eps = (
            None
        )
        hidden_states_222 = torch.nn.functional.layer_norm(
            hidden_states_221,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_bias_,
            item_43,
        )
        l_self_modules_vision_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_bias_ = (item_43) = (
            None
        )
        hidden_states_223 = torch._C._nn.pad(
            hidden_states_222, (0, 0, 0, 6, 0, 6), "constant", None
        )
        hidden_states_222 = None
        hidden_states_224 = hidden_states_223.reshape(1, 5, 14, 5, 14, 1280)
        hidden_states_223 = None
        permute_166 = hidden_states_224.permute(0, 1, 3, 2, 4, 5)
        hidden_states_224 = None
        contiguous_57 = permute_166.contiguous()
        permute_166 = None
        windows_19 = contiguous_57.reshape(-1, 14, 14, 1280)
        contiguous_57 = None
        linear_84 = torch._C._nn.linear(
            windows_19,
            l_self_modules_vision_encoder_modules_layers_modules_21_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_21_modules_attn_modules_qkv_parameters_bias_,
        )
        windows_19 = l_self_modules_vision_encoder_modules_layers_modules_21_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_21_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_267 = linear_84.reshape(25, 196, 3, 16, -1)
        linear_84 = None
        qkv_21 = reshape_267.permute(2, 0, 3, 1, 4)
        reshape_267 = None
        reshape_268 = qkv_21.reshape(3, 400, 196, -1)
        qkv_21 = None
        unbind_21 = reshape_268.unbind(0)
        reshape_268 = None
        query_42 = unbind_21[0]
        key_42 = unbind_21[1]
        value_42 = unbind_21[2]
        unbind_21 = None
        reshape_269 = l_self_modules_vision_encoder_modules_layers_modules_21_modules_attn_parameters_rel_pos_h_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_21_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_168 = reshape_269.permute(0, 2, 1)
        reshape_269 = None
        rel_pos_resized_84 = torch.nn.functional.interpolate(
            permute_168, size=27, mode="linear"
        )
        permute_168 = None
        reshape_270 = rel_pos_resized_84.reshape(-1, 27)
        rel_pos_resized_84 = None
        rel_pos_resized_85 = reshape_270.permute(1, 0)
        reshape_270 = None
        arange_84 = torch.arange(14)
        getitem_261 = arange_84[(slice(None, None, None), None)]
        arange_84 = None
        q_coords_42 = getitem_261 * 1.0
        getitem_261 = None
        arange_85 = torch.arange(14)
        getitem_262 = arange_85[(None, slice(None, None, None))]
        arange_85 = None
        k_coords_42 = getitem_262 * 1.0
        getitem_262 = None
        sub_45 = q_coords_42 - k_coords_42
        q_coords_42 = k_coords_42 = None
        relative_coords_42 = sub_45 + 13.0
        sub_45 = None
        long_42 = relative_coords_42.long()
        relative_coords_42 = None
        relative_position_height_21 = rel_pos_resized_85[long_42]
        rel_pos_resized_85 = long_42 = None
        reshape_271 = l_self_modules_vision_encoder_modules_layers_modules_21_modules_attn_parameters_rel_pos_w_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_21_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_170 = reshape_271.permute(0, 2, 1)
        reshape_271 = None
        rel_pos_resized_86 = torch.nn.functional.interpolate(
            permute_170, size=27, mode="linear"
        )
        permute_170 = None
        reshape_272 = rel_pos_resized_86.reshape(-1, 27)
        rel_pos_resized_86 = None
        rel_pos_resized_87 = reshape_272.permute(1, 0)
        reshape_272 = None
        arange_86 = torch.arange(14)
        getitem_264 = arange_86[(slice(None, None, None), None)]
        arange_86 = None
        q_coords_43 = getitem_264 * 1.0
        getitem_264 = None
        arange_87 = torch.arange(14)
        getitem_265 = arange_87[(None, slice(None, None, None))]
        arange_87 = None
        k_coords_43 = getitem_265 * 1.0
        getitem_265 = None
        sub_46 = q_coords_43 - k_coords_43
        q_coords_43 = k_coords_43 = None
        relative_coords_43 = sub_46 + 13.0
        sub_46 = None
        long_43 = relative_coords_43.long()
        relative_coords_43 = None
        relative_position_width_21 = rel_pos_resized_87[long_43]
        rel_pos_resized_87 = long_43 = None
        reshaped_query_21 = query_42.reshape(400, 14, 14, 80)
        rel_h_21 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_21, relative_position_height_21
        )
        relative_position_height_21 = None
        rel_w_21 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_21, relative_position_width_21
        )
        reshaped_query_21 = relative_position_width_21 = None
        getitem_267 = rel_h_21[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_21 = None
        getitem_268 = rel_w_21[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_21 = None
        decomposed_rel_pos_42 = getitem_267 + getitem_268
        getitem_267 = getitem_268 = None
        decomposed_rel_pos_43 = decomposed_rel_pos_42.reshape(25, 16, 196, 196)
        decomposed_rel_pos_42 = None
        query_43 = query_42.view(25, 16, 196, -1)
        query_42 = None
        key_43 = key_42.view(25, 16, 196, -1)
        key_42 = None
        value_43 = value_42.view(25, 16, 196, -1)
        value_42 = None
        attn_output_63 = torch._C._nn.scaled_dot_product_attention(
            query_43, key_43, value_43, attn_mask=decomposed_rel_pos_43
        )
        query_43 = key_43 = value_43 = decomposed_rel_pos_43 = None
        view_87 = attn_output_63.view(25, 16, 14, 14, -1)
        attn_output_63 = None
        permute_172 = view_87.permute(0, 2, 3, 1, 4)
        view_87 = None
        attn_output_64 = permute_172.reshape(25, 14, 14, -1)
        permute_172 = None
        attn_output_65 = torch._C._nn.linear(
            attn_output_64,
            l_self_modules_vision_encoder_modules_layers_modules_21_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_21_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_64 = l_self_modules_vision_encoder_modules_layers_modules_21_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_21_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_225 = attn_output_65.reshape(1, 5, 5, 14, 14, -1)
        attn_output_65 = None
        permute_173 = hidden_states_225.permute(0, 1, 3, 2, 4, 5)
        hidden_states_225 = None
        contiguous_58 = permute_173.contiguous()
        permute_173 = None
        hidden_states_226 = contiguous_58.reshape(1, 70, 70, -1)
        contiguous_58 = None
        getitem_269 = hidden_states_226[
            (
                slice(None, None, None),
                slice(None, 64, None),
                slice(None, 64, None),
                slice(None, None, None),
            )
        ]
        hidden_states_226 = None
        hidden_states_227 = getitem_269.contiguous()
        getitem_269 = None
        hidden_states_228 = hidden_states_221 + hidden_states_227
        hidden_states_221 = hidden_states_227 = None
        item_44 = (
            l_self_modules_vision_encoder_modules_layers_modules_21_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_21_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_21 = torch.nn.functional.layer_norm(
            hidden_states_228,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_bias_,
            item_44,
        )
        l_self_modules_vision_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_bias_ = (item_44) = (
            None
        )
        hidden_states_229 = torch._C._nn.linear(
            layernorm_output_21,
            l_self_modules_vision_encoder_modules_layers_modules_21_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_21_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_21 = l_self_modules_vision_encoder_modules_layers_modules_21_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_21_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_230 = torch._C._nn.gelu(hidden_states_229)
        hidden_states_229 = None
        hidden_states_231 = torch._C._nn.linear(
            hidden_states_230,
            l_self_modules_vision_encoder_modules_layers_modules_21_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_21_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_230 = l_self_modules_vision_encoder_modules_layers_modules_21_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_21_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_232 = hidden_states_228 + hidden_states_231
        hidden_states_228 = hidden_states_231 = None
        item_45 = (
            l_self_modules_vision_encoder_modules_layers_modules_22_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_22_modules_layer_norm1_eps = (
            None
        )
        hidden_states_233 = torch.nn.functional.layer_norm(
            hidden_states_232,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_bias_,
            item_45,
        )
        l_self_modules_vision_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_bias_ = (item_45) = (
            None
        )
        hidden_states_234 = torch._C._nn.pad(
            hidden_states_233, (0, 0, 0, 6, 0, 6), "constant", None
        )
        hidden_states_233 = None
        hidden_states_235 = hidden_states_234.reshape(1, 5, 14, 5, 14, 1280)
        hidden_states_234 = None
        permute_174 = hidden_states_235.permute(0, 1, 3, 2, 4, 5)
        hidden_states_235 = None
        contiguous_60 = permute_174.contiguous()
        permute_174 = None
        windows_20 = contiguous_60.reshape(-1, 14, 14, 1280)
        contiguous_60 = None
        linear_88 = torch._C._nn.linear(
            windows_20,
            l_self_modules_vision_encoder_modules_layers_modules_22_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_22_modules_attn_modules_qkv_parameters_bias_,
        )
        windows_20 = l_self_modules_vision_encoder_modules_layers_modules_22_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_22_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_280 = linear_88.reshape(25, 196, 3, 16, -1)
        linear_88 = None
        qkv_22 = reshape_280.permute(2, 0, 3, 1, 4)
        reshape_280 = None
        reshape_281 = qkv_22.reshape(3, 400, 196, -1)
        qkv_22 = None
        unbind_22 = reshape_281.unbind(0)
        reshape_281 = None
        query_44 = unbind_22[0]
        key_44 = unbind_22[1]
        value_44 = unbind_22[2]
        unbind_22 = None
        reshape_282 = l_self_modules_vision_encoder_modules_layers_modules_22_modules_attn_parameters_rel_pos_h_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_22_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_176 = reshape_282.permute(0, 2, 1)
        reshape_282 = None
        rel_pos_resized_88 = torch.nn.functional.interpolate(
            permute_176, size=27, mode="linear"
        )
        permute_176 = None
        reshape_283 = rel_pos_resized_88.reshape(-1, 27)
        rel_pos_resized_88 = None
        rel_pos_resized_89 = reshape_283.permute(1, 0)
        reshape_283 = None
        arange_88 = torch.arange(14)
        getitem_273 = arange_88[(slice(None, None, None), None)]
        arange_88 = None
        q_coords_44 = getitem_273 * 1.0
        getitem_273 = None
        arange_89 = torch.arange(14)
        getitem_274 = arange_89[(None, slice(None, None, None))]
        arange_89 = None
        k_coords_44 = getitem_274 * 1.0
        getitem_274 = None
        sub_47 = q_coords_44 - k_coords_44
        q_coords_44 = k_coords_44 = None
        relative_coords_44 = sub_47 + 13.0
        sub_47 = None
        long_44 = relative_coords_44.long()
        relative_coords_44 = None
        relative_position_height_22 = rel_pos_resized_89[long_44]
        rel_pos_resized_89 = long_44 = None
        reshape_284 = l_self_modules_vision_encoder_modules_layers_modules_22_modules_attn_parameters_rel_pos_w_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_22_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_178 = reshape_284.permute(0, 2, 1)
        reshape_284 = None
        rel_pos_resized_90 = torch.nn.functional.interpolate(
            permute_178, size=27, mode="linear"
        )
        permute_178 = None
        reshape_285 = rel_pos_resized_90.reshape(-1, 27)
        rel_pos_resized_90 = None
        rel_pos_resized_91 = reshape_285.permute(1, 0)
        reshape_285 = None
        arange_90 = torch.arange(14)
        getitem_276 = arange_90[(slice(None, None, None), None)]
        arange_90 = None
        q_coords_45 = getitem_276 * 1.0
        getitem_276 = None
        arange_91 = torch.arange(14)
        getitem_277 = arange_91[(None, slice(None, None, None))]
        arange_91 = None
        k_coords_45 = getitem_277 * 1.0
        getitem_277 = None
        sub_48 = q_coords_45 - k_coords_45
        q_coords_45 = k_coords_45 = None
        relative_coords_45 = sub_48 + 13.0
        sub_48 = None
        long_45 = relative_coords_45.long()
        relative_coords_45 = None
        relative_position_width_22 = rel_pos_resized_91[long_45]
        rel_pos_resized_91 = long_45 = None
        reshaped_query_22 = query_44.reshape(400, 14, 14, 80)
        rel_h_22 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_22, relative_position_height_22
        )
        relative_position_height_22 = None
        rel_w_22 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_22, relative_position_width_22
        )
        reshaped_query_22 = relative_position_width_22 = None
        getitem_279 = rel_h_22[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_22 = None
        getitem_280 = rel_w_22[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_22 = None
        decomposed_rel_pos_44 = getitem_279 + getitem_280
        getitem_279 = getitem_280 = None
        decomposed_rel_pos_45 = decomposed_rel_pos_44.reshape(25, 16, 196, 196)
        decomposed_rel_pos_44 = None
        query_45 = query_44.view(25, 16, 196, -1)
        query_44 = None
        key_45 = key_44.view(25, 16, 196, -1)
        key_44 = None
        value_45 = value_44.view(25, 16, 196, -1)
        value_44 = None
        attn_output_66 = torch._C._nn.scaled_dot_product_attention(
            query_45, key_45, value_45, attn_mask=decomposed_rel_pos_45
        )
        query_45 = key_45 = value_45 = decomposed_rel_pos_45 = None
        view_91 = attn_output_66.view(25, 16, 14, 14, -1)
        attn_output_66 = None
        permute_180 = view_91.permute(0, 2, 3, 1, 4)
        view_91 = None
        attn_output_67 = permute_180.reshape(25, 14, 14, -1)
        permute_180 = None
        attn_output_68 = torch._C._nn.linear(
            attn_output_67,
            l_self_modules_vision_encoder_modules_layers_modules_22_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_22_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_67 = l_self_modules_vision_encoder_modules_layers_modules_22_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_22_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_236 = attn_output_68.reshape(1, 5, 5, 14, 14, -1)
        attn_output_68 = None
        permute_181 = hidden_states_236.permute(0, 1, 3, 2, 4, 5)
        hidden_states_236 = None
        contiguous_61 = permute_181.contiguous()
        permute_181 = None
        hidden_states_237 = contiguous_61.reshape(1, 70, 70, -1)
        contiguous_61 = None
        getitem_281 = hidden_states_237[
            (
                slice(None, None, None),
                slice(None, 64, None),
                slice(None, 64, None),
                slice(None, None, None),
            )
        ]
        hidden_states_237 = None
        hidden_states_238 = getitem_281.contiguous()
        getitem_281 = None
        hidden_states_239 = hidden_states_232 + hidden_states_238
        hidden_states_232 = hidden_states_238 = None
        item_46 = (
            l_self_modules_vision_encoder_modules_layers_modules_22_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_22_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_22 = torch.nn.functional.layer_norm(
            hidden_states_239,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_bias_,
            item_46,
        )
        l_self_modules_vision_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_bias_ = (item_46) = (
            None
        )
        hidden_states_240 = torch._C._nn.linear(
            layernorm_output_22,
            l_self_modules_vision_encoder_modules_layers_modules_22_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_22_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_22 = l_self_modules_vision_encoder_modules_layers_modules_22_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_22_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_241 = torch._C._nn.gelu(hidden_states_240)
        hidden_states_240 = None
        hidden_states_242 = torch._C._nn.linear(
            hidden_states_241,
            l_self_modules_vision_encoder_modules_layers_modules_22_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_22_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_241 = l_self_modules_vision_encoder_modules_layers_modules_22_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_22_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_243 = hidden_states_239 + hidden_states_242
        hidden_states_239 = hidden_states_242 = None
        item_47 = (
            l_self_modules_vision_encoder_modules_layers_modules_23_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_23_modules_layer_norm1_eps = (
            None
        )
        hidden_states_244 = torch.nn.functional.layer_norm(
            hidden_states_243,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_bias_,
            item_47,
        )
        l_self_modules_vision_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_bias_ = (item_47) = (
            None
        )
        linear_92 = torch._C._nn.linear(
            hidden_states_244,
            l_self_modules_vision_encoder_modules_layers_modules_23_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_23_modules_attn_modules_qkv_parameters_bias_,
        )
        hidden_states_244 = l_self_modules_vision_encoder_modules_layers_modules_23_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_23_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_291 = linear_92.reshape(1, 4096, 3, 16, -1)
        linear_92 = None
        qkv_23 = reshape_291.permute(2, 0, 3, 1, 4)
        reshape_291 = None
        reshape_292 = qkv_23.reshape(3, 16, 4096, -1)
        qkv_23 = None
        unbind_23 = reshape_292.unbind(0)
        reshape_292 = None
        query_46 = unbind_23[0]
        key_46 = unbind_23[1]
        value_46 = unbind_23[2]
        unbind_23 = None
        reshape_293 = l_self_modules_vision_encoder_modules_layers_modules_23_modules_attn_parameters_rel_pos_h_.reshape(
            1, 127, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_23_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_183 = reshape_293.permute(0, 2, 1)
        reshape_293 = None
        rel_pos_resized_92 = torch.nn.functional.interpolate(
            permute_183, size=127, mode="linear"
        )
        permute_183 = None
        reshape_294 = rel_pos_resized_92.reshape(-1, 127)
        rel_pos_resized_92 = None
        rel_pos_resized_93 = reshape_294.permute(1, 0)
        reshape_294 = None
        arange_92 = torch.arange(64)
        getitem_285 = arange_92[(slice(None, None, None), None)]
        arange_92 = None
        q_coords_46 = getitem_285 * 1.0
        getitem_285 = None
        arange_93 = torch.arange(64)
        getitem_286 = arange_93[(None, slice(None, None, None))]
        arange_93 = None
        k_coords_46 = getitem_286 * 1.0
        getitem_286 = None
        sub_49 = q_coords_46 - k_coords_46
        q_coords_46 = k_coords_46 = None
        relative_coords_46 = sub_49 + 63.0
        sub_49 = None
        long_46 = relative_coords_46.long()
        relative_coords_46 = None
        relative_position_height_23 = rel_pos_resized_93[long_46]
        rel_pos_resized_93 = long_46 = None
        reshape_295 = l_self_modules_vision_encoder_modules_layers_modules_23_modules_attn_parameters_rel_pos_w_.reshape(
            1, 127, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_23_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_185 = reshape_295.permute(0, 2, 1)
        reshape_295 = None
        rel_pos_resized_94 = torch.nn.functional.interpolate(
            permute_185, size=127, mode="linear"
        )
        permute_185 = None
        reshape_296 = rel_pos_resized_94.reshape(-1, 127)
        rel_pos_resized_94 = None
        rel_pos_resized_95 = reshape_296.permute(1, 0)
        reshape_296 = None
        arange_94 = torch.arange(64)
        getitem_288 = arange_94[(slice(None, None, None), None)]
        arange_94 = None
        q_coords_47 = getitem_288 * 1.0
        getitem_288 = None
        arange_95 = torch.arange(64)
        getitem_289 = arange_95[(None, slice(None, None, None))]
        arange_95 = None
        k_coords_47 = getitem_289 * 1.0
        getitem_289 = None
        sub_50 = q_coords_47 - k_coords_47
        q_coords_47 = k_coords_47 = None
        relative_coords_47 = sub_50 + 63.0
        sub_50 = None
        long_47 = relative_coords_47.long()
        relative_coords_47 = None
        relative_position_width_23 = rel_pos_resized_95[long_47]
        rel_pos_resized_95 = long_47 = None
        reshaped_query_23 = query_46.reshape(16, 64, 64, 80)
        rel_h_23 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_23, relative_position_height_23
        )
        relative_position_height_23 = None
        rel_w_23 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_23, relative_position_width_23
        )
        reshaped_query_23 = relative_position_width_23 = None
        getitem_291 = rel_h_23[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_23 = None
        getitem_292 = rel_w_23[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_23 = None
        decomposed_rel_pos_46 = getitem_291 + getitem_292
        getitem_291 = getitem_292 = None
        decomposed_rel_pos_47 = decomposed_rel_pos_46.reshape(1, 16, 4096, 4096)
        decomposed_rel_pos_46 = None
        query_47 = query_46.view(1, 16, 4096, -1)
        query_46 = None
        key_47 = key_46.view(1, 16, 4096, -1)
        key_46 = None
        value_47 = value_46.view(1, 16, 4096, -1)
        value_46 = None
        attn_output_69 = torch._C._nn.scaled_dot_product_attention(
            query_47, key_47, value_47, attn_mask=decomposed_rel_pos_47
        )
        query_47 = key_47 = value_47 = decomposed_rel_pos_47 = None
        view_95 = attn_output_69.view(1, 16, 64, 64, -1)
        attn_output_69 = None
        permute_187 = view_95.permute(0, 2, 3, 1, 4)
        view_95 = None
        attn_output_70 = permute_187.reshape(1, 64, 64, -1)
        permute_187 = None
        attn_output_71 = torch._C._nn.linear(
            attn_output_70,
            l_self_modules_vision_encoder_modules_layers_modules_23_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_23_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_70 = l_self_modules_vision_encoder_modules_layers_modules_23_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_23_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_245 = hidden_states_243 + attn_output_71
        hidden_states_243 = attn_output_71 = None
        item_48 = (
            l_self_modules_vision_encoder_modules_layers_modules_23_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_23_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_23 = torch.nn.functional.layer_norm(
            hidden_states_245,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_bias_,
            item_48,
        )
        l_self_modules_vision_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_bias_ = (item_48) = (
            None
        )
        hidden_states_246 = torch._C._nn.linear(
            layernorm_output_23,
            l_self_modules_vision_encoder_modules_layers_modules_23_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_23_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_23 = l_self_modules_vision_encoder_modules_layers_modules_23_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_23_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_247 = torch._C._nn.gelu(hidden_states_246)
        hidden_states_246 = None
        hidden_states_248 = torch._C._nn.linear(
            hidden_states_247,
            l_self_modules_vision_encoder_modules_layers_modules_23_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_23_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_247 = l_self_modules_vision_encoder_modules_layers_modules_23_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_23_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_249 = hidden_states_245 + hidden_states_248
        hidden_states_245 = hidden_states_248 = None
        item_49 = (
            l_self_modules_vision_encoder_modules_layers_modules_24_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_24_modules_layer_norm1_eps = (
            None
        )
        hidden_states_250 = torch.nn.functional.layer_norm(
            hidden_states_249,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_24_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_24_modules_layer_norm1_parameters_bias_,
            item_49,
        )
        l_self_modules_vision_encoder_modules_layers_modules_24_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_24_modules_layer_norm1_parameters_bias_ = (item_49) = (
            None
        )
        hidden_states_251 = torch._C._nn.pad(
            hidden_states_250, (0, 0, 0, 6, 0, 6), "constant", None
        )
        hidden_states_250 = None
        hidden_states_252 = hidden_states_251.reshape(1, 5, 14, 5, 14, 1280)
        hidden_states_251 = None
        permute_188 = hidden_states_252.permute(0, 1, 3, 2, 4, 5)
        hidden_states_252 = None
        contiguous_63 = permute_188.contiguous()
        permute_188 = None
        windows_21 = contiguous_63.reshape(-1, 14, 14, 1280)
        contiguous_63 = None
        linear_96 = torch._C._nn.linear(
            windows_21,
            l_self_modules_vision_encoder_modules_layers_modules_24_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_24_modules_attn_modules_qkv_parameters_bias_,
        )
        windows_21 = l_self_modules_vision_encoder_modules_layers_modules_24_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_24_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_302 = linear_96.reshape(25, 196, 3, 16, -1)
        linear_96 = None
        qkv_24 = reshape_302.permute(2, 0, 3, 1, 4)
        reshape_302 = None
        reshape_303 = qkv_24.reshape(3, 400, 196, -1)
        qkv_24 = None
        unbind_24 = reshape_303.unbind(0)
        reshape_303 = None
        query_48 = unbind_24[0]
        key_48 = unbind_24[1]
        value_48 = unbind_24[2]
        unbind_24 = None
        reshape_304 = l_self_modules_vision_encoder_modules_layers_modules_24_modules_attn_parameters_rel_pos_h_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_24_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_190 = reshape_304.permute(0, 2, 1)
        reshape_304 = None
        rel_pos_resized_96 = torch.nn.functional.interpolate(
            permute_190, size=27, mode="linear"
        )
        permute_190 = None
        reshape_305 = rel_pos_resized_96.reshape(-1, 27)
        rel_pos_resized_96 = None
        rel_pos_resized_97 = reshape_305.permute(1, 0)
        reshape_305 = None
        arange_96 = torch.arange(14)
        getitem_296 = arange_96[(slice(None, None, None), None)]
        arange_96 = None
        q_coords_48 = getitem_296 * 1.0
        getitem_296 = None
        arange_97 = torch.arange(14)
        getitem_297 = arange_97[(None, slice(None, None, None))]
        arange_97 = None
        k_coords_48 = getitem_297 * 1.0
        getitem_297 = None
        sub_51 = q_coords_48 - k_coords_48
        q_coords_48 = k_coords_48 = None
        relative_coords_48 = sub_51 + 13.0
        sub_51 = None
        long_48 = relative_coords_48.long()
        relative_coords_48 = None
        relative_position_height_24 = rel_pos_resized_97[long_48]
        rel_pos_resized_97 = long_48 = None
        reshape_306 = l_self_modules_vision_encoder_modules_layers_modules_24_modules_attn_parameters_rel_pos_w_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_24_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_192 = reshape_306.permute(0, 2, 1)
        reshape_306 = None
        rel_pos_resized_98 = torch.nn.functional.interpolate(
            permute_192, size=27, mode="linear"
        )
        permute_192 = None
        reshape_307 = rel_pos_resized_98.reshape(-1, 27)
        rel_pos_resized_98 = None
        rel_pos_resized_99 = reshape_307.permute(1, 0)
        reshape_307 = None
        arange_98 = torch.arange(14)
        getitem_299 = arange_98[(slice(None, None, None), None)]
        arange_98 = None
        q_coords_49 = getitem_299 * 1.0
        getitem_299 = None
        arange_99 = torch.arange(14)
        getitem_300 = arange_99[(None, slice(None, None, None))]
        arange_99 = None
        k_coords_49 = getitem_300 * 1.0
        getitem_300 = None
        sub_52 = q_coords_49 - k_coords_49
        q_coords_49 = k_coords_49 = None
        relative_coords_49 = sub_52 + 13.0
        sub_52 = None
        long_49 = relative_coords_49.long()
        relative_coords_49 = None
        relative_position_width_24 = rel_pos_resized_99[long_49]
        rel_pos_resized_99 = long_49 = None
        reshaped_query_24 = query_48.reshape(400, 14, 14, 80)
        rel_h_24 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_24, relative_position_height_24
        )
        relative_position_height_24 = None
        rel_w_24 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_24, relative_position_width_24
        )
        reshaped_query_24 = relative_position_width_24 = None
        getitem_302 = rel_h_24[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_24 = None
        getitem_303 = rel_w_24[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_24 = None
        decomposed_rel_pos_48 = getitem_302 + getitem_303
        getitem_302 = getitem_303 = None
        decomposed_rel_pos_49 = decomposed_rel_pos_48.reshape(25, 16, 196, 196)
        decomposed_rel_pos_48 = None
        query_49 = query_48.view(25, 16, 196, -1)
        query_48 = None
        key_49 = key_48.view(25, 16, 196, -1)
        key_48 = None
        value_49 = value_48.view(25, 16, 196, -1)
        value_48 = None
        attn_output_72 = torch._C._nn.scaled_dot_product_attention(
            query_49, key_49, value_49, attn_mask=decomposed_rel_pos_49
        )
        query_49 = key_49 = value_49 = decomposed_rel_pos_49 = None
        view_99 = attn_output_72.view(25, 16, 14, 14, -1)
        attn_output_72 = None
        permute_194 = view_99.permute(0, 2, 3, 1, 4)
        view_99 = None
        attn_output_73 = permute_194.reshape(25, 14, 14, -1)
        permute_194 = None
        attn_output_74 = torch._C._nn.linear(
            attn_output_73,
            l_self_modules_vision_encoder_modules_layers_modules_24_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_24_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_73 = l_self_modules_vision_encoder_modules_layers_modules_24_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_24_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_253 = attn_output_74.reshape(1, 5, 5, 14, 14, -1)
        attn_output_74 = None
        permute_195 = hidden_states_253.permute(0, 1, 3, 2, 4, 5)
        hidden_states_253 = None
        contiguous_64 = permute_195.contiguous()
        permute_195 = None
        hidden_states_254 = contiguous_64.reshape(1, 70, 70, -1)
        contiguous_64 = None
        getitem_304 = hidden_states_254[
            (
                slice(None, None, None),
                slice(None, 64, None),
                slice(None, 64, None),
                slice(None, None, None),
            )
        ]
        hidden_states_254 = None
        hidden_states_255 = getitem_304.contiguous()
        getitem_304 = None
        hidden_states_256 = hidden_states_249 + hidden_states_255
        hidden_states_249 = hidden_states_255 = None
        item_50 = (
            l_self_modules_vision_encoder_modules_layers_modules_24_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_24_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_24 = torch.nn.functional.layer_norm(
            hidden_states_256,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_24_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_24_modules_layer_norm2_parameters_bias_,
            item_50,
        )
        l_self_modules_vision_encoder_modules_layers_modules_24_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_24_modules_layer_norm2_parameters_bias_ = (item_50) = (
            None
        )
        hidden_states_257 = torch._C._nn.linear(
            layernorm_output_24,
            l_self_modules_vision_encoder_modules_layers_modules_24_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_24_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_24 = l_self_modules_vision_encoder_modules_layers_modules_24_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_24_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_258 = torch._C._nn.gelu(hidden_states_257)
        hidden_states_257 = None
        hidden_states_259 = torch._C._nn.linear(
            hidden_states_258,
            l_self_modules_vision_encoder_modules_layers_modules_24_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_24_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_258 = l_self_modules_vision_encoder_modules_layers_modules_24_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_24_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_260 = hidden_states_256 + hidden_states_259
        hidden_states_256 = hidden_states_259 = None
        item_51 = (
            l_self_modules_vision_encoder_modules_layers_modules_25_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_25_modules_layer_norm1_eps = (
            None
        )
        hidden_states_261 = torch.nn.functional.layer_norm(
            hidden_states_260,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_25_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_25_modules_layer_norm1_parameters_bias_,
            item_51,
        )
        l_self_modules_vision_encoder_modules_layers_modules_25_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_25_modules_layer_norm1_parameters_bias_ = (item_51) = (
            None
        )
        hidden_states_262 = torch._C._nn.pad(
            hidden_states_261, (0, 0, 0, 6, 0, 6), "constant", None
        )
        hidden_states_261 = None
        hidden_states_263 = hidden_states_262.reshape(1, 5, 14, 5, 14, 1280)
        hidden_states_262 = None
        permute_196 = hidden_states_263.permute(0, 1, 3, 2, 4, 5)
        hidden_states_263 = None
        contiguous_66 = permute_196.contiguous()
        permute_196 = None
        windows_22 = contiguous_66.reshape(-1, 14, 14, 1280)
        contiguous_66 = None
        linear_100 = torch._C._nn.linear(
            windows_22,
            l_self_modules_vision_encoder_modules_layers_modules_25_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_25_modules_attn_modules_qkv_parameters_bias_,
        )
        windows_22 = l_self_modules_vision_encoder_modules_layers_modules_25_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_25_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_315 = linear_100.reshape(25, 196, 3, 16, -1)
        linear_100 = None
        qkv_25 = reshape_315.permute(2, 0, 3, 1, 4)
        reshape_315 = None
        reshape_316 = qkv_25.reshape(3, 400, 196, -1)
        qkv_25 = None
        unbind_25 = reshape_316.unbind(0)
        reshape_316 = None
        query_50 = unbind_25[0]
        key_50 = unbind_25[1]
        value_50 = unbind_25[2]
        unbind_25 = None
        reshape_317 = l_self_modules_vision_encoder_modules_layers_modules_25_modules_attn_parameters_rel_pos_h_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_25_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_198 = reshape_317.permute(0, 2, 1)
        reshape_317 = None
        rel_pos_resized_100 = torch.nn.functional.interpolate(
            permute_198, size=27, mode="linear"
        )
        permute_198 = None
        reshape_318 = rel_pos_resized_100.reshape(-1, 27)
        rel_pos_resized_100 = None
        rel_pos_resized_101 = reshape_318.permute(1, 0)
        reshape_318 = None
        arange_100 = torch.arange(14)
        getitem_308 = arange_100[(slice(None, None, None), None)]
        arange_100 = None
        q_coords_50 = getitem_308 * 1.0
        getitem_308 = None
        arange_101 = torch.arange(14)
        getitem_309 = arange_101[(None, slice(None, None, None))]
        arange_101 = None
        k_coords_50 = getitem_309 * 1.0
        getitem_309 = None
        sub_53 = q_coords_50 - k_coords_50
        q_coords_50 = k_coords_50 = None
        relative_coords_50 = sub_53 + 13.0
        sub_53 = None
        long_50 = relative_coords_50.long()
        relative_coords_50 = None
        relative_position_height_25 = rel_pos_resized_101[long_50]
        rel_pos_resized_101 = long_50 = None
        reshape_319 = l_self_modules_vision_encoder_modules_layers_modules_25_modules_attn_parameters_rel_pos_w_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_25_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_200 = reshape_319.permute(0, 2, 1)
        reshape_319 = None
        rel_pos_resized_102 = torch.nn.functional.interpolate(
            permute_200, size=27, mode="linear"
        )
        permute_200 = None
        reshape_320 = rel_pos_resized_102.reshape(-1, 27)
        rel_pos_resized_102 = None
        rel_pos_resized_103 = reshape_320.permute(1, 0)
        reshape_320 = None
        arange_102 = torch.arange(14)
        getitem_311 = arange_102[(slice(None, None, None), None)]
        arange_102 = None
        q_coords_51 = getitem_311 * 1.0
        getitem_311 = None
        arange_103 = torch.arange(14)
        getitem_312 = arange_103[(None, slice(None, None, None))]
        arange_103 = None
        k_coords_51 = getitem_312 * 1.0
        getitem_312 = None
        sub_54 = q_coords_51 - k_coords_51
        q_coords_51 = k_coords_51 = None
        relative_coords_51 = sub_54 + 13.0
        sub_54 = None
        long_51 = relative_coords_51.long()
        relative_coords_51 = None
        relative_position_width_25 = rel_pos_resized_103[long_51]
        rel_pos_resized_103 = long_51 = None
        reshaped_query_25 = query_50.reshape(400, 14, 14, 80)
        rel_h_25 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_25, relative_position_height_25
        )
        relative_position_height_25 = None
        rel_w_25 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_25, relative_position_width_25
        )
        reshaped_query_25 = relative_position_width_25 = None
        getitem_314 = rel_h_25[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_25 = None
        getitem_315 = rel_w_25[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_25 = None
        decomposed_rel_pos_50 = getitem_314 + getitem_315
        getitem_314 = getitem_315 = None
        decomposed_rel_pos_51 = decomposed_rel_pos_50.reshape(25, 16, 196, 196)
        decomposed_rel_pos_50 = None
        query_51 = query_50.view(25, 16, 196, -1)
        query_50 = None
        key_51 = key_50.view(25, 16, 196, -1)
        key_50 = None
        value_51 = value_50.view(25, 16, 196, -1)
        value_50 = None
        attn_output_75 = torch._C._nn.scaled_dot_product_attention(
            query_51, key_51, value_51, attn_mask=decomposed_rel_pos_51
        )
        query_51 = key_51 = value_51 = decomposed_rel_pos_51 = None
        view_103 = attn_output_75.view(25, 16, 14, 14, -1)
        attn_output_75 = None
        permute_202 = view_103.permute(0, 2, 3, 1, 4)
        view_103 = None
        attn_output_76 = permute_202.reshape(25, 14, 14, -1)
        permute_202 = None
        attn_output_77 = torch._C._nn.linear(
            attn_output_76,
            l_self_modules_vision_encoder_modules_layers_modules_25_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_25_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_76 = l_self_modules_vision_encoder_modules_layers_modules_25_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_25_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_264 = attn_output_77.reshape(1, 5, 5, 14, 14, -1)
        attn_output_77 = None
        permute_203 = hidden_states_264.permute(0, 1, 3, 2, 4, 5)
        hidden_states_264 = None
        contiguous_67 = permute_203.contiguous()
        permute_203 = None
        hidden_states_265 = contiguous_67.reshape(1, 70, 70, -1)
        contiguous_67 = None
        getitem_316 = hidden_states_265[
            (
                slice(None, None, None),
                slice(None, 64, None),
                slice(None, 64, None),
                slice(None, None, None),
            )
        ]
        hidden_states_265 = None
        hidden_states_266 = getitem_316.contiguous()
        getitem_316 = None
        hidden_states_267 = hidden_states_260 + hidden_states_266
        hidden_states_260 = hidden_states_266 = None
        item_52 = (
            l_self_modules_vision_encoder_modules_layers_modules_25_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_25_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_25 = torch.nn.functional.layer_norm(
            hidden_states_267,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_25_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_25_modules_layer_norm2_parameters_bias_,
            item_52,
        )
        l_self_modules_vision_encoder_modules_layers_modules_25_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_25_modules_layer_norm2_parameters_bias_ = (item_52) = (
            None
        )
        hidden_states_268 = torch._C._nn.linear(
            layernorm_output_25,
            l_self_modules_vision_encoder_modules_layers_modules_25_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_25_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_25 = l_self_modules_vision_encoder_modules_layers_modules_25_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_25_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_269 = torch._C._nn.gelu(hidden_states_268)
        hidden_states_268 = None
        hidden_states_270 = torch._C._nn.linear(
            hidden_states_269,
            l_self_modules_vision_encoder_modules_layers_modules_25_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_25_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_269 = l_self_modules_vision_encoder_modules_layers_modules_25_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_25_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_271 = hidden_states_267 + hidden_states_270
        hidden_states_267 = hidden_states_270 = None
        item_53 = (
            l_self_modules_vision_encoder_modules_layers_modules_26_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_26_modules_layer_norm1_eps = (
            None
        )
        hidden_states_272 = torch.nn.functional.layer_norm(
            hidden_states_271,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_26_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_26_modules_layer_norm1_parameters_bias_,
            item_53,
        )
        l_self_modules_vision_encoder_modules_layers_modules_26_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_26_modules_layer_norm1_parameters_bias_ = (item_53) = (
            None
        )
        hidden_states_273 = torch._C._nn.pad(
            hidden_states_272, (0, 0, 0, 6, 0, 6), "constant", None
        )
        hidden_states_272 = None
        hidden_states_274 = hidden_states_273.reshape(1, 5, 14, 5, 14, 1280)
        hidden_states_273 = None
        permute_204 = hidden_states_274.permute(0, 1, 3, 2, 4, 5)
        hidden_states_274 = None
        contiguous_69 = permute_204.contiguous()
        permute_204 = None
        windows_23 = contiguous_69.reshape(-1, 14, 14, 1280)
        contiguous_69 = None
        linear_104 = torch._C._nn.linear(
            windows_23,
            l_self_modules_vision_encoder_modules_layers_modules_26_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_26_modules_attn_modules_qkv_parameters_bias_,
        )
        windows_23 = l_self_modules_vision_encoder_modules_layers_modules_26_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_26_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_328 = linear_104.reshape(25, 196, 3, 16, -1)
        linear_104 = None
        qkv_26 = reshape_328.permute(2, 0, 3, 1, 4)
        reshape_328 = None
        reshape_329 = qkv_26.reshape(3, 400, 196, -1)
        qkv_26 = None
        unbind_26 = reshape_329.unbind(0)
        reshape_329 = None
        query_52 = unbind_26[0]
        key_52 = unbind_26[1]
        value_52 = unbind_26[2]
        unbind_26 = None
        reshape_330 = l_self_modules_vision_encoder_modules_layers_modules_26_modules_attn_parameters_rel_pos_h_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_26_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_206 = reshape_330.permute(0, 2, 1)
        reshape_330 = None
        rel_pos_resized_104 = torch.nn.functional.interpolate(
            permute_206, size=27, mode="linear"
        )
        permute_206 = None
        reshape_331 = rel_pos_resized_104.reshape(-1, 27)
        rel_pos_resized_104 = None
        rel_pos_resized_105 = reshape_331.permute(1, 0)
        reshape_331 = None
        arange_104 = torch.arange(14)
        getitem_320 = arange_104[(slice(None, None, None), None)]
        arange_104 = None
        q_coords_52 = getitem_320 * 1.0
        getitem_320 = None
        arange_105 = torch.arange(14)
        getitem_321 = arange_105[(None, slice(None, None, None))]
        arange_105 = None
        k_coords_52 = getitem_321 * 1.0
        getitem_321 = None
        sub_55 = q_coords_52 - k_coords_52
        q_coords_52 = k_coords_52 = None
        relative_coords_52 = sub_55 + 13.0
        sub_55 = None
        long_52 = relative_coords_52.long()
        relative_coords_52 = None
        relative_position_height_26 = rel_pos_resized_105[long_52]
        rel_pos_resized_105 = long_52 = None
        reshape_332 = l_self_modules_vision_encoder_modules_layers_modules_26_modules_attn_parameters_rel_pos_w_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_26_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_208 = reshape_332.permute(0, 2, 1)
        reshape_332 = None
        rel_pos_resized_106 = torch.nn.functional.interpolate(
            permute_208, size=27, mode="linear"
        )
        permute_208 = None
        reshape_333 = rel_pos_resized_106.reshape(-1, 27)
        rel_pos_resized_106 = None
        rel_pos_resized_107 = reshape_333.permute(1, 0)
        reshape_333 = None
        arange_106 = torch.arange(14)
        getitem_323 = arange_106[(slice(None, None, None), None)]
        arange_106 = None
        q_coords_53 = getitem_323 * 1.0
        getitem_323 = None
        arange_107 = torch.arange(14)
        getitem_324 = arange_107[(None, slice(None, None, None))]
        arange_107 = None
        k_coords_53 = getitem_324 * 1.0
        getitem_324 = None
        sub_56 = q_coords_53 - k_coords_53
        q_coords_53 = k_coords_53 = None
        relative_coords_53 = sub_56 + 13.0
        sub_56 = None
        long_53 = relative_coords_53.long()
        relative_coords_53 = None
        relative_position_width_26 = rel_pos_resized_107[long_53]
        rel_pos_resized_107 = long_53 = None
        reshaped_query_26 = query_52.reshape(400, 14, 14, 80)
        rel_h_26 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_26, relative_position_height_26
        )
        relative_position_height_26 = None
        rel_w_26 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_26, relative_position_width_26
        )
        reshaped_query_26 = relative_position_width_26 = None
        getitem_326 = rel_h_26[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_26 = None
        getitem_327 = rel_w_26[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_26 = None
        decomposed_rel_pos_52 = getitem_326 + getitem_327
        getitem_326 = getitem_327 = None
        decomposed_rel_pos_53 = decomposed_rel_pos_52.reshape(25, 16, 196, 196)
        decomposed_rel_pos_52 = None
        query_53 = query_52.view(25, 16, 196, -1)
        query_52 = None
        key_53 = key_52.view(25, 16, 196, -1)
        key_52 = None
        value_53 = value_52.view(25, 16, 196, -1)
        value_52 = None
        attn_output_78 = torch._C._nn.scaled_dot_product_attention(
            query_53, key_53, value_53, attn_mask=decomposed_rel_pos_53
        )
        query_53 = key_53 = value_53 = decomposed_rel_pos_53 = None
        view_107 = attn_output_78.view(25, 16, 14, 14, -1)
        attn_output_78 = None
        permute_210 = view_107.permute(0, 2, 3, 1, 4)
        view_107 = None
        attn_output_79 = permute_210.reshape(25, 14, 14, -1)
        permute_210 = None
        attn_output_80 = torch._C._nn.linear(
            attn_output_79,
            l_self_modules_vision_encoder_modules_layers_modules_26_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_26_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_79 = l_self_modules_vision_encoder_modules_layers_modules_26_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_26_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_275 = attn_output_80.reshape(1, 5, 5, 14, 14, -1)
        attn_output_80 = None
        permute_211 = hidden_states_275.permute(0, 1, 3, 2, 4, 5)
        hidden_states_275 = None
        contiguous_70 = permute_211.contiguous()
        permute_211 = None
        hidden_states_276 = contiguous_70.reshape(1, 70, 70, -1)
        contiguous_70 = None
        getitem_328 = hidden_states_276[
            (
                slice(None, None, None),
                slice(None, 64, None),
                slice(None, 64, None),
                slice(None, None, None),
            )
        ]
        hidden_states_276 = None
        hidden_states_277 = getitem_328.contiguous()
        getitem_328 = None
        hidden_states_278 = hidden_states_271 + hidden_states_277
        hidden_states_271 = hidden_states_277 = None
        item_54 = (
            l_self_modules_vision_encoder_modules_layers_modules_26_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_26_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_26 = torch.nn.functional.layer_norm(
            hidden_states_278,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_26_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_26_modules_layer_norm2_parameters_bias_,
            item_54,
        )
        l_self_modules_vision_encoder_modules_layers_modules_26_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_26_modules_layer_norm2_parameters_bias_ = (item_54) = (
            None
        )
        hidden_states_279 = torch._C._nn.linear(
            layernorm_output_26,
            l_self_modules_vision_encoder_modules_layers_modules_26_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_26_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_26 = l_self_modules_vision_encoder_modules_layers_modules_26_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_26_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_280 = torch._C._nn.gelu(hidden_states_279)
        hidden_states_279 = None
        hidden_states_281 = torch._C._nn.linear(
            hidden_states_280,
            l_self_modules_vision_encoder_modules_layers_modules_26_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_26_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_280 = l_self_modules_vision_encoder_modules_layers_modules_26_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_26_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_282 = hidden_states_278 + hidden_states_281
        hidden_states_278 = hidden_states_281 = None
        item_55 = (
            l_self_modules_vision_encoder_modules_layers_modules_27_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_27_modules_layer_norm1_eps = (
            None
        )
        hidden_states_283 = torch.nn.functional.layer_norm(
            hidden_states_282,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_27_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_27_modules_layer_norm1_parameters_bias_,
            item_55,
        )
        l_self_modules_vision_encoder_modules_layers_modules_27_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_27_modules_layer_norm1_parameters_bias_ = (item_55) = (
            None
        )
        hidden_states_284 = torch._C._nn.pad(
            hidden_states_283, (0, 0, 0, 6, 0, 6), "constant", None
        )
        hidden_states_283 = None
        hidden_states_285 = hidden_states_284.reshape(1, 5, 14, 5, 14, 1280)
        hidden_states_284 = None
        permute_212 = hidden_states_285.permute(0, 1, 3, 2, 4, 5)
        hidden_states_285 = None
        contiguous_72 = permute_212.contiguous()
        permute_212 = None
        windows_24 = contiguous_72.reshape(-1, 14, 14, 1280)
        contiguous_72 = None
        linear_108 = torch._C._nn.linear(
            windows_24,
            l_self_modules_vision_encoder_modules_layers_modules_27_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_27_modules_attn_modules_qkv_parameters_bias_,
        )
        windows_24 = l_self_modules_vision_encoder_modules_layers_modules_27_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_27_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_341 = linear_108.reshape(25, 196, 3, 16, -1)
        linear_108 = None
        qkv_27 = reshape_341.permute(2, 0, 3, 1, 4)
        reshape_341 = None
        reshape_342 = qkv_27.reshape(3, 400, 196, -1)
        qkv_27 = None
        unbind_27 = reshape_342.unbind(0)
        reshape_342 = None
        query_54 = unbind_27[0]
        key_54 = unbind_27[1]
        value_54 = unbind_27[2]
        unbind_27 = None
        reshape_343 = l_self_modules_vision_encoder_modules_layers_modules_27_modules_attn_parameters_rel_pos_h_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_27_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_214 = reshape_343.permute(0, 2, 1)
        reshape_343 = None
        rel_pos_resized_108 = torch.nn.functional.interpolate(
            permute_214, size=27, mode="linear"
        )
        permute_214 = None
        reshape_344 = rel_pos_resized_108.reshape(-1, 27)
        rel_pos_resized_108 = None
        rel_pos_resized_109 = reshape_344.permute(1, 0)
        reshape_344 = None
        arange_108 = torch.arange(14)
        getitem_332 = arange_108[(slice(None, None, None), None)]
        arange_108 = None
        q_coords_54 = getitem_332 * 1.0
        getitem_332 = None
        arange_109 = torch.arange(14)
        getitem_333 = arange_109[(None, slice(None, None, None))]
        arange_109 = None
        k_coords_54 = getitem_333 * 1.0
        getitem_333 = None
        sub_57 = q_coords_54 - k_coords_54
        q_coords_54 = k_coords_54 = None
        relative_coords_54 = sub_57 + 13.0
        sub_57 = None
        long_54 = relative_coords_54.long()
        relative_coords_54 = None
        relative_position_height_27 = rel_pos_resized_109[long_54]
        rel_pos_resized_109 = long_54 = None
        reshape_345 = l_self_modules_vision_encoder_modules_layers_modules_27_modules_attn_parameters_rel_pos_w_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_27_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_216 = reshape_345.permute(0, 2, 1)
        reshape_345 = None
        rel_pos_resized_110 = torch.nn.functional.interpolate(
            permute_216, size=27, mode="linear"
        )
        permute_216 = None
        reshape_346 = rel_pos_resized_110.reshape(-1, 27)
        rel_pos_resized_110 = None
        rel_pos_resized_111 = reshape_346.permute(1, 0)
        reshape_346 = None
        arange_110 = torch.arange(14)
        getitem_335 = arange_110[(slice(None, None, None), None)]
        arange_110 = None
        q_coords_55 = getitem_335 * 1.0
        getitem_335 = None
        arange_111 = torch.arange(14)
        getitem_336 = arange_111[(None, slice(None, None, None))]
        arange_111 = None
        k_coords_55 = getitem_336 * 1.0
        getitem_336 = None
        sub_58 = q_coords_55 - k_coords_55
        q_coords_55 = k_coords_55 = None
        relative_coords_55 = sub_58 + 13.0
        sub_58 = None
        long_55 = relative_coords_55.long()
        relative_coords_55 = None
        relative_position_width_27 = rel_pos_resized_111[long_55]
        rel_pos_resized_111 = long_55 = None
        reshaped_query_27 = query_54.reshape(400, 14, 14, 80)
        rel_h_27 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_27, relative_position_height_27
        )
        relative_position_height_27 = None
        rel_w_27 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_27, relative_position_width_27
        )
        reshaped_query_27 = relative_position_width_27 = None
        getitem_338 = rel_h_27[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_27 = None
        getitem_339 = rel_w_27[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_27 = None
        decomposed_rel_pos_54 = getitem_338 + getitem_339
        getitem_338 = getitem_339 = None
        decomposed_rel_pos_55 = decomposed_rel_pos_54.reshape(25, 16, 196, 196)
        decomposed_rel_pos_54 = None
        query_55 = query_54.view(25, 16, 196, -1)
        query_54 = None
        key_55 = key_54.view(25, 16, 196, -1)
        key_54 = None
        value_55 = value_54.view(25, 16, 196, -1)
        value_54 = None
        attn_output_81 = torch._C._nn.scaled_dot_product_attention(
            query_55, key_55, value_55, attn_mask=decomposed_rel_pos_55
        )
        query_55 = key_55 = value_55 = decomposed_rel_pos_55 = None
        view_111 = attn_output_81.view(25, 16, 14, 14, -1)
        attn_output_81 = None
        permute_218 = view_111.permute(0, 2, 3, 1, 4)
        view_111 = None
        attn_output_82 = permute_218.reshape(25, 14, 14, -1)
        permute_218 = None
        attn_output_83 = torch._C._nn.linear(
            attn_output_82,
            l_self_modules_vision_encoder_modules_layers_modules_27_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_27_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_82 = l_self_modules_vision_encoder_modules_layers_modules_27_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_27_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_286 = attn_output_83.reshape(1, 5, 5, 14, 14, -1)
        attn_output_83 = None
        permute_219 = hidden_states_286.permute(0, 1, 3, 2, 4, 5)
        hidden_states_286 = None
        contiguous_73 = permute_219.contiguous()
        permute_219 = None
        hidden_states_287 = contiguous_73.reshape(1, 70, 70, -1)
        contiguous_73 = None
        getitem_340 = hidden_states_287[
            (
                slice(None, None, None),
                slice(None, 64, None),
                slice(None, 64, None),
                slice(None, None, None),
            )
        ]
        hidden_states_287 = None
        hidden_states_288 = getitem_340.contiguous()
        getitem_340 = None
        hidden_states_289 = hidden_states_282 + hidden_states_288
        hidden_states_282 = hidden_states_288 = None
        item_56 = (
            l_self_modules_vision_encoder_modules_layers_modules_27_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_27_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_27 = torch.nn.functional.layer_norm(
            hidden_states_289,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_27_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_27_modules_layer_norm2_parameters_bias_,
            item_56,
        )
        l_self_modules_vision_encoder_modules_layers_modules_27_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_27_modules_layer_norm2_parameters_bias_ = (item_56) = (
            None
        )
        hidden_states_290 = torch._C._nn.linear(
            layernorm_output_27,
            l_self_modules_vision_encoder_modules_layers_modules_27_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_27_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_27 = l_self_modules_vision_encoder_modules_layers_modules_27_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_27_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_291 = torch._C._nn.gelu(hidden_states_290)
        hidden_states_290 = None
        hidden_states_292 = torch._C._nn.linear(
            hidden_states_291,
            l_self_modules_vision_encoder_modules_layers_modules_27_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_27_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_291 = l_self_modules_vision_encoder_modules_layers_modules_27_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_27_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_293 = hidden_states_289 + hidden_states_292
        hidden_states_289 = hidden_states_292 = None
        item_57 = (
            l_self_modules_vision_encoder_modules_layers_modules_28_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_28_modules_layer_norm1_eps = (
            None
        )
        hidden_states_294 = torch.nn.functional.layer_norm(
            hidden_states_293,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_28_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_28_modules_layer_norm1_parameters_bias_,
            item_57,
        )
        l_self_modules_vision_encoder_modules_layers_modules_28_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_28_modules_layer_norm1_parameters_bias_ = (item_57) = (
            None
        )
        hidden_states_295 = torch._C._nn.pad(
            hidden_states_294, (0, 0, 0, 6, 0, 6), "constant", None
        )
        hidden_states_294 = None
        hidden_states_296 = hidden_states_295.reshape(1, 5, 14, 5, 14, 1280)
        hidden_states_295 = None
        permute_220 = hidden_states_296.permute(0, 1, 3, 2, 4, 5)
        hidden_states_296 = None
        contiguous_75 = permute_220.contiguous()
        permute_220 = None
        windows_25 = contiguous_75.reshape(-1, 14, 14, 1280)
        contiguous_75 = None
        linear_112 = torch._C._nn.linear(
            windows_25,
            l_self_modules_vision_encoder_modules_layers_modules_28_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_28_modules_attn_modules_qkv_parameters_bias_,
        )
        windows_25 = l_self_modules_vision_encoder_modules_layers_modules_28_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_28_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_354 = linear_112.reshape(25, 196, 3, 16, -1)
        linear_112 = None
        qkv_28 = reshape_354.permute(2, 0, 3, 1, 4)
        reshape_354 = None
        reshape_355 = qkv_28.reshape(3, 400, 196, -1)
        qkv_28 = None
        unbind_28 = reshape_355.unbind(0)
        reshape_355 = None
        query_56 = unbind_28[0]
        key_56 = unbind_28[1]
        value_56 = unbind_28[2]
        unbind_28 = None
        reshape_356 = l_self_modules_vision_encoder_modules_layers_modules_28_modules_attn_parameters_rel_pos_h_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_28_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_222 = reshape_356.permute(0, 2, 1)
        reshape_356 = None
        rel_pos_resized_112 = torch.nn.functional.interpolate(
            permute_222, size=27, mode="linear"
        )
        permute_222 = None
        reshape_357 = rel_pos_resized_112.reshape(-1, 27)
        rel_pos_resized_112 = None
        rel_pos_resized_113 = reshape_357.permute(1, 0)
        reshape_357 = None
        arange_112 = torch.arange(14)
        getitem_344 = arange_112[(slice(None, None, None), None)]
        arange_112 = None
        q_coords_56 = getitem_344 * 1.0
        getitem_344 = None
        arange_113 = torch.arange(14)
        getitem_345 = arange_113[(None, slice(None, None, None))]
        arange_113 = None
        k_coords_56 = getitem_345 * 1.0
        getitem_345 = None
        sub_59 = q_coords_56 - k_coords_56
        q_coords_56 = k_coords_56 = None
        relative_coords_56 = sub_59 + 13.0
        sub_59 = None
        long_56 = relative_coords_56.long()
        relative_coords_56 = None
        relative_position_height_28 = rel_pos_resized_113[long_56]
        rel_pos_resized_113 = long_56 = None
        reshape_358 = l_self_modules_vision_encoder_modules_layers_modules_28_modules_attn_parameters_rel_pos_w_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_28_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_224 = reshape_358.permute(0, 2, 1)
        reshape_358 = None
        rel_pos_resized_114 = torch.nn.functional.interpolate(
            permute_224, size=27, mode="linear"
        )
        permute_224 = None
        reshape_359 = rel_pos_resized_114.reshape(-1, 27)
        rel_pos_resized_114 = None
        rel_pos_resized_115 = reshape_359.permute(1, 0)
        reshape_359 = None
        arange_114 = torch.arange(14)
        getitem_347 = arange_114[(slice(None, None, None), None)]
        arange_114 = None
        q_coords_57 = getitem_347 * 1.0
        getitem_347 = None
        arange_115 = torch.arange(14)
        getitem_348 = arange_115[(None, slice(None, None, None))]
        arange_115 = None
        k_coords_57 = getitem_348 * 1.0
        getitem_348 = None
        sub_60 = q_coords_57 - k_coords_57
        q_coords_57 = k_coords_57 = None
        relative_coords_57 = sub_60 + 13.0
        sub_60 = None
        long_57 = relative_coords_57.long()
        relative_coords_57 = None
        relative_position_width_28 = rel_pos_resized_115[long_57]
        rel_pos_resized_115 = long_57 = None
        reshaped_query_28 = query_56.reshape(400, 14, 14, 80)
        rel_h_28 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_28, relative_position_height_28
        )
        relative_position_height_28 = None
        rel_w_28 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_28, relative_position_width_28
        )
        reshaped_query_28 = relative_position_width_28 = None
        getitem_350 = rel_h_28[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_28 = None
        getitem_351 = rel_w_28[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_28 = None
        decomposed_rel_pos_56 = getitem_350 + getitem_351
        getitem_350 = getitem_351 = None
        decomposed_rel_pos_57 = decomposed_rel_pos_56.reshape(25, 16, 196, 196)
        decomposed_rel_pos_56 = None
        query_57 = query_56.view(25, 16, 196, -1)
        query_56 = None
        key_57 = key_56.view(25, 16, 196, -1)
        key_56 = None
        value_57 = value_56.view(25, 16, 196, -1)
        value_56 = None
        attn_output_84 = torch._C._nn.scaled_dot_product_attention(
            query_57, key_57, value_57, attn_mask=decomposed_rel_pos_57
        )
        query_57 = key_57 = value_57 = decomposed_rel_pos_57 = None
        view_115 = attn_output_84.view(25, 16, 14, 14, -1)
        attn_output_84 = None
        permute_226 = view_115.permute(0, 2, 3, 1, 4)
        view_115 = None
        attn_output_85 = permute_226.reshape(25, 14, 14, -1)
        permute_226 = None
        attn_output_86 = torch._C._nn.linear(
            attn_output_85,
            l_self_modules_vision_encoder_modules_layers_modules_28_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_28_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_85 = l_self_modules_vision_encoder_modules_layers_modules_28_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_28_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_297 = attn_output_86.reshape(1, 5, 5, 14, 14, -1)
        attn_output_86 = None
        permute_227 = hidden_states_297.permute(0, 1, 3, 2, 4, 5)
        hidden_states_297 = None
        contiguous_76 = permute_227.contiguous()
        permute_227 = None
        hidden_states_298 = contiguous_76.reshape(1, 70, 70, -1)
        contiguous_76 = None
        getitem_352 = hidden_states_298[
            (
                slice(None, None, None),
                slice(None, 64, None),
                slice(None, 64, None),
                slice(None, None, None),
            )
        ]
        hidden_states_298 = None
        hidden_states_299 = getitem_352.contiguous()
        getitem_352 = None
        hidden_states_300 = hidden_states_293 + hidden_states_299
        hidden_states_293 = hidden_states_299 = None
        item_58 = (
            l_self_modules_vision_encoder_modules_layers_modules_28_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_28_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_28 = torch.nn.functional.layer_norm(
            hidden_states_300,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_28_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_28_modules_layer_norm2_parameters_bias_,
            item_58,
        )
        l_self_modules_vision_encoder_modules_layers_modules_28_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_28_modules_layer_norm2_parameters_bias_ = (item_58) = (
            None
        )
        hidden_states_301 = torch._C._nn.linear(
            layernorm_output_28,
            l_self_modules_vision_encoder_modules_layers_modules_28_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_28_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_28 = l_self_modules_vision_encoder_modules_layers_modules_28_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_28_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_302 = torch._C._nn.gelu(hidden_states_301)
        hidden_states_301 = None
        hidden_states_303 = torch._C._nn.linear(
            hidden_states_302,
            l_self_modules_vision_encoder_modules_layers_modules_28_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_28_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_302 = l_self_modules_vision_encoder_modules_layers_modules_28_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_28_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_304 = hidden_states_300 + hidden_states_303
        hidden_states_300 = hidden_states_303 = None
        item_59 = (
            l_self_modules_vision_encoder_modules_layers_modules_29_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_29_modules_layer_norm1_eps = (
            None
        )
        hidden_states_305 = torch.nn.functional.layer_norm(
            hidden_states_304,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_29_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_29_modules_layer_norm1_parameters_bias_,
            item_59,
        )
        l_self_modules_vision_encoder_modules_layers_modules_29_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_29_modules_layer_norm1_parameters_bias_ = (item_59) = (
            None
        )
        hidden_states_306 = torch._C._nn.pad(
            hidden_states_305, (0, 0, 0, 6, 0, 6), "constant", None
        )
        hidden_states_305 = None
        hidden_states_307 = hidden_states_306.reshape(1, 5, 14, 5, 14, 1280)
        hidden_states_306 = None
        permute_228 = hidden_states_307.permute(0, 1, 3, 2, 4, 5)
        hidden_states_307 = None
        contiguous_78 = permute_228.contiguous()
        permute_228 = None
        windows_26 = contiguous_78.reshape(-1, 14, 14, 1280)
        contiguous_78 = None
        linear_116 = torch._C._nn.linear(
            windows_26,
            l_self_modules_vision_encoder_modules_layers_modules_29_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_29_modules_attn_modules_qkv_parameters_bias_,
        )
        windows_26 = l_self_modules_vision_encoder_modules_layers_modules_29_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_29_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_367 = linear_116.reshape(25, 196, 3, 16, -1)
        linear_116 = None
        qkv_29 = reshape_367.permute(2, 0, 3, 1, 4)
        reshape_367 = None
        reshape_368 = qkv_29.reshape(3, 400, 196, -1)
        qkv_29 = None
        unbind_29 = reshape_368.unbind(0)
        reshape_368 = None
        query_58 = unbind_29[0]
        key_58 = unbind_29[1]
        value_58 = unbind_29[2]
        unbind_29 = None
        reshape_369 = l_self_modules_vision_encoder_modules_layers_modules_29_modules_attn_parameters_rel_pos_h_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_29_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_230 = reshape_369.permute(0, 2, 1)
        reshape_369 = None
        rel_pos_resized_116 = torch.nn.functional.interpolate(
            permute_230, size=27, mode="linear"
        )
        permute_230 = None
        reshape_370 = rel_pos_resized_116.reshape(-1, 27)
        rel_pos_resized_116 = None
        rel_pos_resized_117 = reshape_370.permute(1, 0)
        reshape_370 = None
        arange_116 = torch.arange(14)
        getitem_356 = arange_116[(slice(None, None, None), None)]
        arange_116 = None
        q_coords_58 = getitem_356 * 1.0
        getitem_356 = None
        arange_117 = torch.arange(14)
        getitem_357 = arange_117[(None, slice(None, None, None))]
        arange_117 = None
        k_coords_58 = getitem_357 * 1.0
        getitem_357 = None
        sub_61 = q_coords_58 - k_coords_58
        q_coords_58 = k_coords_58 = None
        relative_coords_58 = sub_61 + 13.0
        sub_61 = None
        long_58 = relative_coords_58.long()
        relative_coords_58 = None
        relative_position_height_29 = rel_pos_resized_117[long_58]
        rel_pos_resized_117 = long_58 = None
        reshape_371 = l_self_modules_vision_encoder_modules_layers_modules_29_modules_attn_parameters_rel_pos_w_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_29_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_232 = reshape_371.permute(0, 2, 1)
        reshape_371 = None
        rel_pos_resized_118 = torch.nn.functional.interpolate(
            permute_232, size=27, mode="linear"
        )
        permute_232 = None
        reshape_372 = rel_pos_resized_118.reshape(-1, 27)
        rel_pos_resized_118 = None
        rel_pos_resized_119 = reshape_372.permute(1, 0)
        reshape_372 = None
        arange_118 = torch.arange(14)
        getitem_359 = arange_118[(slice(None, None, None), None)]
        arange_118 = None
        q_coords_59 = getitem_359 * 1.0
        getitem_359 = None
        arange_119 = torch.arange(14)
        getitem_360 = arange_119[(None, slice(None, None, None))]
        arange_119 = None
        k_coords_59 = getitem_360 * 1.0
        getitem_360 = None
        sub_62 = q_coords_59 - k_coords_59
        q_coords_59 = k_coords_59 = None
        relative_coords_59 = sub_62 + 13.0
        sub_62 = None
        long_59 = relative_coords_59.long()
        relative_coords_59 = None
        relative_position_width_29 = rel_pos_resized_119[long_59]
        rel_pos_resized_119 = long_59 = None
        reshaped_query_29 = query_58.reshape(400, 14, 14, 80)
        rel_h_29 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_29, relative_position_height_29
        )
        relative_position_height_29 = None
        rel_w_29 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_29, relative_position_width_29
        )
        reshaped_query_29 = relative_position_width_29 = None
        getitem_362 = rel_h_29[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_29 = None
        getitem_363 = rel_w_29[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_29 = None
        decomposed_rel_pos_58 = getitem_362 + getitem_363
        getitem_362 = getitem_363 = None
        decomposed_rel_pos_59 = decomposed_rel_pos_58.reshape(25, 16, 196, 196)
        decomposed_rel_pos_58 = None
        query_59 = query_58.view(25, 16, 196, -1)
        query_58 = None
        key_59 = key_58.view(25, 16, 196, -1)
        key_58 = None
        value_59 = value_58.view(25, 16, 196, -1)
        value_58 = None
        attn_output_87 = torch._C._nn.scaled_dot_product_attention(
            query_59, key_59, value_59, attn_mask=decomposed_rel_pos_59
        )
        query_59 = key_59 = value_59 = decomposed_rel_pos_59 = None
        view_119 = attn_output_87.view(25, 16, 14, 14, -1)
        attn_output_87 = None
        permute_234 = view_119.permute(0, 2, 3, 1, 4)
        view_119 = None
        attn_output_88 = permute_234.reshape(25, 14, 14, -1)
        permute_234 = None
        attn_output_89 = torch._C._nn.linear(
            attn_output_88,
            l_self_modules_vision_encoder_modules_layers_modules_29_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_29_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_88 = l_self_modules_vision_encoder_modules_layers_modules_29_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_29_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_308 = attn_output_89.reshape(1, 5, 5, 14, 14, -1)
        attn_output_89 = None
        permute_235 = hidden_states_308.permute(0, 1, 3, 2, 4, 5)
        hidden_states_308 = None
        contiguous_79 = permute_235.contiguous()
        permute_235 = None
        hidden_states_309 = contiguous_79.reshape(1, 70, 70, -1)
        contiguous_79 = None
        getitem_364 = hidden_states_309[
            (
                slice(None, None, None),
                slice(None, 64, None),
                slice(None, 64, None),
                slice(None, None, None),
            )
        ]
        hidden_states_309 = None
        hidden_states_310 = getitem_364.contiguous()
        getitem_364 = None
        hidden_states_311 = hidden_states_304 + hidden_states_310
        hidden_states_304 = hidden_states_310 = None
        item_60 = (
            l_self_modules_vision_encoder_modules_layers_modules_29_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_29_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_29 = torch.nn.functional.layer_norm(
            hidden_states_311,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_29_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_29_modules_layer_norm2_parameters_bias_,
            item_60,
        )
        l_self_modules_vision_encoder_modules_layers_modules_29_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_29_modules_layer_norm2_parameters_bias_ = (item_60) = (
            None
        )
        hidden_states_312 = torch._C._nn.linear(
            layernorm_output_29,
            l_self_modules_vision_encoder_modules_layers_modules_29_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_29_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_29 = l_self_modules_vision_encoder_modules_layers_modules_29_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_29_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_313 = torch._C._nn.gelu(hidden_states_312)
        hidden_states_312 = None
        hidden_states_314 = torch._C._nn.linear(
            hidden_states_313,
            l_self_modules_vision_encoder_modules_layers_modules_29_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_29_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_313 = l_self_modules_vision_encoder_modules_layers_modules_29_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_29_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_315 = hidden_states_311 + hidden_states_314
        hidden_states_311 = hidden_states_314 = None
        item_61 = (
            l_self_modules_vision_encoder_modules_layers_modules_30_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_30_modules_layer_norm1_eps = (
            None
        )
        hidden_states_316 = torch.nn.functional.layer_norm(
            hidden_states_315,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_30_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_30_modules_layer_norm1_parameters_bias_,
            item_61,
        )
        l_self_modules_vision_encoder_modules_layers_modules_30_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_30_modules_layer_norm1_parameters_bias_ = (item_61) = (
            None
        )
        hidden_states_317 = torch._C._nn.pad(
            hidden_states_316, (0, 0, 0, 6, 0, 6), "constant", None
        )
        hidden_states_316 = None
        hidden_states_318 = hidden_states_317.reshape(1, 5, 14, 5, 14, 1280)
        hidden_states_317 = None
        permute_236 = hidden_states_318.permute(0, 1, 3, 2, 4, 5)
        hidden_states_318 = None
        contiguous_81 = permute_236.contiguous()
        permute_236 = None
        windows_27 = contiguous_81.reshape(-1, 14, 14, 1280)
        contiguous_81 = None
        linear_120 = torch._C._nn.linear(
            windows_27,
            l_self_modules_vision_encoder_modules_layers_modules_30_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_30_modules_attn_modules_qkv_parameters_bias_,
        )
        windows_27 = l_self_modules_vision_encoder_modules_layers_modules_30_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_30_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_380 = linear_120.reshape(25, 196, 3, 16, -1)
        linear_120 = None
        qkv_30 = reshape_380.permute(2, 0, 3, 1, 4)
        reshape_380 = None
        reshape_381 = qkv_30.reshape(3, 400, 196, -1)
        qkv_30 = None
        unbind_30 = reshape_381.unbind(0)
        reshape_381 = None
        query_60 = unbind_30[0]
        key_60 = unbind_30[1]
        value_60 = unbind_30[2]
        unbind_30 = None
        reshape_382 = l_self_modules_vision_encoder_modules_layers_modules_30_modules_attn_parameters_rel_pos_h_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_30_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_238 = reshape_382.permute(0, 2, 1)
        reshape_382 = None
        rel_pos_resized_120 = torch.nn.functional.interpolate(
            permute_238, size=27, mode="linear"
        )
        permute_238 = None
        reshape_383 = rel_pos_resized_120.reshape(-1, 27)
        rel_pos_resized_120 = None
        rel_pos_resized_121 = reshape_383.permute(1, 0)
        reshape_383 = None
        arange_120 = torch.arange(14)
        getitem_368 = arange_120[(slice(None, None, None), None)]
        arange_120 = None
        q_coords_60 = getitem_368 * 1.0
        getitem_368 = None
        arange_121 = torch.arange(14)
        getitem_369 = arange_121[(None, slice(None, None, None))]
        arange_121 = None
        k_coords_60 = getitem_369 * 1.0
        getitem_369 = None
        sub_63 = q_coords_60 - k_coords_60
        q_coords_60 = k_coords_60 = None
        relative_coords_60 = sub_63 + 13.0
        sub_63 = None
        long_60 = relative_coords_60.long()
        relative_coords_60 = None
        relative_position_height_30 = rel_pos_resized_121[long_60]
        rel_pos_resized_121 = long_60 = None
        reshape_384 = l_self_modules_vision_encoder_modules_layers_modules_30_modules_attn_parameters_rel_pos_w_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_30_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_240 = reshape_384.permute(0, 2, 1)
        reshape_384 = None
        rel_pos_resized_122 = torch.nn.functional.interpolate(
            permute_240, size=27, mode="linear"
        )
        permute_240 = None
        reshape_385 = rel_pos_resized_122.reshape(-1, 27)
        rel_pos_resized_122 = None
        rel_pos_resized_123 = reshape_385.permute(1, 0)
        reshape_385 = None
        arange_122 = torch.arange(14)
        getitem_371 = arange_122[(slice(None, None, None), None)]
        arange_122 = None
        q_coords_61 = getitem_371 * 1.0
        getitem_371 = None
        arange_123 = torch.arange(14)
        getitem_372 = arange_123[(None, slice(None, None, None))]
        arange_123 = None
        k_coords_61 = getitem_372 * 1.0
        getitem_372 = None
        sub_64 = q_coords_61 - k_coords_61
        q_coords_61 = k_coords_61 = None
        relative_coords_61 = sub_64 + 13.0
        sub_64 = None
        long_61 = relative_coords_61.long()
        relative_coords_61 = None
        relative_position_width_30 = rel_pos_resized_123[long_61]
        rel_pos_resized_123 = long_61 = None
        reshaped_query_30 = query_60.reshape(400, 14, 14, 80)
        rel_h_30 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_30, relative_position_height_30
        )
        relative_position_height_30 = None
        rel_w_30 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_30, relative_position_width_30
        )
        reshaped_query_30 = relative_position_width_30 = None
        getitem_374 = rel_h_30[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_30 = None
        getitem_375 = rel_w_30[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_30 = None
        decomposed_rel_pos_60 = getitem_374 + getitem_375
        getitem_374 = getitem_375 = None
        decomposed_rel_pos_61 = decomposed_rel_pos_60.reshape(25, 16, 196, 196)
        decomposed_rel_pos_60 = None
        query_61 = query_60.view(25, 16, 196, -1)
        query_60 = None
        key_61 = key_60.view(25, 16, 196, -1)
        key_60 = None
        value_61 = value_60.view(25, 16, 196, -1)
        value_60 = None
        attn_output_90 = torch._C._nn.scaled_dot_product_attention(
            query_61, key_61, value_61, attn_mask=decomposed_rel_pos_61
        )
        query_61 = key_61 = value_61 = decomposed_rel_pos_61 = None
        view_123 = attn_output_90.view(25, 16, 14, 14, -1)
        attn_output_90 = None
        permute_242 = view_123.permute(0, 2, 3, 1, 4)
        view_123 = None
        attn_output_91 = permute_242.reshape(25, 14, 14, -1)
        permute_242 = None
        attn_output_92 = torch._C._nn.linear(
            attn_output_91,
            l_self_modules_vision_encoder_modules_layers_modules_30_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_30_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_91 = l_self_modules_vision_encoder_modules_layers_modules_30_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_30_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_319 = attn_output_92.reshape(1, 5, 5, 14, 14, -1)
        attn_output_92 = None
        permute_243 = hidden_states_319.permute(0, 1, 3, 2, 4, 5)
        hidden_states_319 = None
        contiguous_82 = permute_243.contiguous()
        permute_243 = None
        hidden_states_320 = contiguous_82.reshape(1, 70, 70, -1)
        contiguous_82 = None
        getitem_376 = hidden_states_320[
            (
                slice(None, None, None),
                slice(None, 64, None),
                slice(None, 64, None),
                slice(None, None, None),
            )
        ]
        hidden_states_320 = None
        hidden_states_321 = getitem_376.contiguous()
        getitem_376 = None
        hidden_states_322 = hidden_states_315 + hidden_states_321
        hidden_states_315 = hidden_states_321 = None
        item_62 = (
            l_self_modules_vision_encoder_modules_layers_modules_30_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_30_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_30 = torch.nn.functional.layer_norm(
            hidden_states_322,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_30_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_30_modules_layer_norm2_parameters_bias_,
            item_62,
        )
        l_self_modules_vision_encoder_modules_layers_modules_30_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_30_modules_layer_norm2_parameters_bias_ = (item_62) = (
            None
        )
        hidden_states_323 = torch._C._nn.linear(
            layernorm_output_30,
            l_self_modules_vision_encoder_modules_layers_modules_30_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_30_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_30 = l_self_modules_vision_encoder_modules_layers_modules_30_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_30_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_324 = torch._C._nn.gelu(hidden_states_323)
        hidden_states_323 = None
        hidden_states_325 = torch._C._nn.linear(
            hidden_states_324,
            l_self_modules_vision_encoder_modules_layers_modules_30_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_30_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_324 = l_self_modules_vision_encoder_modules_layers_modules_30_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_30_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_326 = hidden_states_322 + hidden_states_325
        hidden_states_322 = hidden_states_325 = None
        item_63 = (
            l_self_modules_vision_encoder_modules_layers_modules_31_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_31_modules_layer_norm1_eps = (
            None
        )
        hidden_states_327 = torch.nn.functional.layer_norm(
            hidden_states_326,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_31_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_31_modules_layer_norm1_parameters_bias_,
            item_63,
        )
        l_self_modules_vision_encoder_modules_layers_modules_31_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_31_modules_layer_norm1_parameters_bias_ = (item_63) = (
            None
        )
        linear_124 = torch._C._nn.linear(
            hidden_states_327,
            l_self_modules_vision_encoder_modules_layers_modules_31_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_31_modules_attn_modules_qkv_parameters_bias_,
        )
        hidden_states_327 = l_self_modules_vision_encoder_modules_layers_modules_31_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_31_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_391 = linear_124.reshape(1, 4096, 3, 16, -1)
        linear_124 = None
        qkv_31 = reshape_391.permute(2, 0, 3, 1, 4)
        reshape_391 = None
        reshape_392 = qkv_31.reshape(3, 16, 4096, -1)
        qkv_31 = None
        unbind_31 = reshape_392.unbind(0)
        reshape_392 = None
        query_62 = unbind_31[0]
        key_62 = unbind_31[1]
        value_62 = unbind_31[2]
        unbind_31 = None
        reshape_393 = l_self_modules_vision_encoder_modules_layers_modules_31_modules_attn_parameters_rel_pos_h_.reshape(
            1, 127, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_31_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_245 = reshape_393.permute(0, 2, 1)
        reshape_393 = None
        rel_pos_resized_124 = torch.nn.functional.interpolate(
            permute_245, size=127, mode="linear"
        )
        permute_245 = None
        reshape_394 = rel_pos_resized_124.reshape(-1, 127)
        rel_pos_resized_124 = None
        rel_pos_resized_125 = reshape_394.permute(1, 0)
        reshape_394 = None
        arange_124 = torch.arange(64)
        getitem_380 = arange_124[(slice(None, None, None), None)]
        arange_124 = None
        q_coords_62 = getitem_380 * 1.0
        getitem_380 = None
        arange_125 = torch.arange(64)
        getitem_381 = arange_125[(None, slice(None, None, None))]
        arange_125 = None
        k_coords_62 = getitem_381 * 1.0
        getitem_381 = None
        sub_65 = q_coords_62 - k_coords_62
        q_coords_62 = k_coords_62 = None
        relative_coords_62 = sub_65 + 63.0
        sub_65 = None
        long_62 = relative_coords_62.long()
        relative_coords_62 = None
        relative_position_height_31 = rel_pos_resized_125[long_62]
        rel_pos_resized_125 = long_62 = None
        reshape_395 = l_self_modules_vision_encoder_modules_layers_modules_31_modules_attn_parameters_rel_pos_w_.reshape(
            1, 127, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_31_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_247 = reshape_395.permute(0, 2, 1)
        reshape_395 = None
        rel_pos_resized_126 = torch.nn.functional.interpolate(
            permute_247, size=127, mode="linear"
        )
        permute_247 = None
        reshape_396 = rel_pos_resized_126.reshape(-1, 127)
        rel_pos_resized_126 = None
        rel_pos_resized_127 = reshape_396.permute(1, 0)
        reshape_396 = None
        arange_126 = torch.arange(64)
        getitem_383 = arange_126[(slice(None, None, None), None)]
        arange_126 = None
        q_coords_63 = getitem_383 * 1.0
        getitem_383 = None
        arange_127 = torch.arange(64)
        getitem_384 = arange_127[(None, slice(None, None, None))]
        arange_127 = None
        k_coords_63 = getitem_384 * 1.0
        getitem_384 = None
        sub_66 = q_coords_63 - k_coords_63
        q_coords_63 = k_coords_63 = None
        relative_coords_63 = sub_66 + 63.0
        sub_66 = None
        long_63 = relative_coords_63.long()
        relative_coords_63 = None
        relative_position_width_31 = rel_pos_resized_127[long_63]
        rel_pos_resized_127 = long_63 = None
        reshaped_query_31 = query_62.reshape(16, 64, 64, 80)
        rel_h_31 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_31, relative_position_height_31
        )
        relative_position_height_31 = None
        rel_w_31 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_31, relative_position_width_31
        )
        reshaped_query_31 = relative_position_width_31 = None
        getitem_386 = rel_h_31[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_31 = None
        getitem_387 = rel_w_31[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_31 = None
        decomposed_rel_pos_62 = getitem_386 + getitem_387
        getitem_386 = getitem_387 = None
        decomposed_rel_pos_63 = decomposed_rel_pos_62.reshape(1, 16, 4096, 4096)
        decomposed_rel_pos_62 = None
        query_63 = query_62.view(1, 16, 4096, -1)
        query_62 = None
        key_63 = key_62.view(1, 16, 4096, -1)
        key_62 = None
        value_63 = value_62.view(1, 16, 4096, -1)
        value_62 = None
        attn_output_93 = torch._C._nn.scaled_dot_product_attention(
            query_63, key_63, value_63, attn_mask=decomposed_rel_pos_63
        )
        query_63 = key_63 = value_63 = decomposed_rel_pos_63 = None
        view_127 = attn_output_93.view(1, 16, 64, 64, -1)
        attn_output_93 = None
        permute_249 = view_127.permute(0, 2, 3, 1, 4)
        view_127 = None
        attn_output_94 = permute_249.reshape(1, 64, 64, -1)
        permute_249 = None
        attn_output_95 = torch._C._nn.linear(
            attn_output_94,
            l_self_modules_vision_encoder_modules_layers_modules_31_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_31_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_94 = l_self_modules_vision_encoder_modules_layers_modules_31_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_31_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_328 = hidden_states_326 + attn_output_95
        hidden_states_326 = attn_output_95 = None
        item_64 = (
            l_self_modules_vision_encoder_modules_layers_modules_31_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_31_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_31 = torch.nn.functional.layer_norm(
            hidden_states_328,
            (1280,),
            l_self_modules_vision_encoder_modules_layers_modules_31_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_31_modules_layer_norm2_parameters_bias_,
            item_64,
        )
        l_self_modules_vision_encoder_modules_layers_modules_31_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_31_modules_layer_norm2_parameters_bias_ = (item_64) = (
            None
        )
        hidden_states_329 = torch._C._nn.linear(
            layernorm_output_31,
            l_self_modules_vision_encoder_modules_layers_modules_31_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_31_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_31 = l_self_modules_vision_encoder_modules_layers_modules_31_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_31_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_330 = torch._C._nn.gelu(hidden_states_329)
        hidden_states_329 = None
        hidden_states_331 = torch._C._nn.linear(
            hidden_states_330,
            l_self_modules_vision_encoder_modules_layers_modules_31_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_31_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_330 = l_self_modules_vision_encoder_modules_layers_modules_31_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_31_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_332 = hidden_states_328 + hidden_states_331
        hidden_states_328 = hidden_states_331 = None
        hidden_states_333 = hidden_states_332.permute(0, 3, 1, 2)
        hidden_states_332 = None
        hidden_states_334 = torch.conv2d(
            hidden_states_333,
            l_self_modules_vision_encoder_modules_neck_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_333 = (
            l_self_modules_vision_encoder_modules_neck_modules_conv1_parameters_weight_
        ) = None
        x = hidden_states_334.float()
        hidden_states_334 = None
        u = x.mean(1, keepdim=True)
        sub_67 = x - u
        pow_1 = sub_67.pow(2)
        sub_67 = None
        s = pow_1.mean(1, keepdim=True)
        pow_1 = None
        sub_68 = x - u
        x = u = None
        item_65 = (
            l_self_modules_vision_encoder_modules_neck_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_neck_modules_layer_norm1_eps = None
        add_161 = s + item_65
        s = item_65 = None
        sqrt = torch.sqrt(add_161)
        add_161 = None
        x_1 = sub_68 / sqrt
        sub_68 = sqrt = None
        x_2 = x_1.to(dtype=torch.float32)
        x_1 = None
        getitem_388 = l_self_modules_vision_encoder_modules_neck_modules_layer_norm1_parameters_weight_[
            (slice(None, None, None), None, None)
        ]
        l_self_modules_vision_encoder_modules_neck_modules_layer_norm1_parameters_weight_ = (
            None
        )
        mul_131 = getitem_388 * x_2
        getitem_388 = x_2 = None
        getitem_389 = l_self_modules_vision_encoder_modules_neck_modules_layer_norm1_parameters_bias_[
            (slice(None, None, None), None, None)
        ]
        l_self_modules_vision_encoder_modules_neck_modules_layer_norm1_parameters_bias_ = (
            None
        )
        x_3 = mul_131 + getitem_389
        mul_131 = getitem_389 = None
        hidden_states_335 = torch.conv2d(
            x_3,
            l_self_modules_vision_encoder_modules_neck_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_3 = (
            l_self_modules_vision_encoder_modules_neck_modules_conv2_parameters_weight_
        ) = None
        x_4 = hidden_states_335.float()
        hidden_states_335 = None
        u_1 = x_4.mean(1, keepdim=True)
        sub_69 = x_4 - u_1
        pow_2 = sub_69.pow(2)
        sub_69 = None
        s_1 = pow_2.mean(1, keepdim=True)
        pow_2 = None
        sub_70 = x_4 - u_1
        x_4 = u_1 = None
        item_66 = (
            l_self_modules_vision_encoder_modules_neck_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_neck_modules_layer_norm2_eps = None
        add_163 = s_1 + item_66
        s_1 = item_66 = None
        sqrt_1 = torch.sqrt(add_163)
        add_163 = None
        x_5 = sub_70 / sqrt_1
        sub_70 = sqrt_1 = None
        x_6 = x_5.to(dtype=torch.float32)
        x_5 = None
        getitem_390 = l_self_modules_vision_encoder_modules_neck_modules_layer_norm2_parameters_weight_[
            (slice(None, None, None), None, None)
        ]
        l_self_modules_vision_encoder_modules_neck_modules_layer_norm2_parameters_weight_ = (
            None
        )
        mul_132 = getitem_390 * x_6
        getitem_390 = x_6 = None
        getitem_391 = l_self_modules_vision_encoder_modules_neck_modules_layer_norm2_parameters_bias_[
            (slice(None, None, None), None, None)
        ]
        l_self_modules_vision_encoder_modules_neck_modules_layer_norm2_parameters_bias_ = (
            None
        )
        x_7 = mul_132 + getitem_391
        mul_132 = getitem_391 = None
        reshape_400 = l_self_modules_prompt_encoder_modules_no_mask_embed_parameters_weight_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_prompt_encoder_modules_no_mask_embed_parameters_weight_ = None
        dense_embeddings = reshape_400.expand(1, -1, 64, 64)
        reshape_400 = None
        output_tokens = torch.cat(
            [
                l_self_modules_mask_decoder_modules_iou_token_parameters_weight_,
                l_self_modules_mask_decoder_modules_mask_tokens_parameters_weight_,
            ],
            dim=0,
        )
        l_self_modules_mask_decoder_modules_iou_token_parameters_weight_ = (
            l_self_modules_mask_decoder_modules_mask_tokens_parameters_weight_
        ) = None
        output_tokens_1 = output_tokens.repeat(1, 1, 1, 1)
        output_tokens = None
        point_embeddings = output_tokens_1.to(torch.float32)
        output_tokens_1 = None
        image_embeddings = x_7 + dense_embeddings
        x_7 = dense_embeddings = None
        image_embeddings_1 = image_embeddings.repeat_interleave(1, 0)
        image_embeddings = None
        image_positional_embeddings_2 = image_positional_embeddings_1.repeat_interleave(
            1, 0
        )
        image_positional_embeddings_1 = None
        flatten = image_embeddings_1.flatten(2)
        image_embeddings_1 = None
        permute_251 = flatten.permute(0, 2, 1)
        flatten = None
        image_embeddings_2 = permute_251.unsqueeze(1)
        permute_251 = None
        flatten_1 = image_positional_embeddings_2.flatten(2)
        image_positional_embeddings_2 = None
        permute_252 = flatten_1.permute(0, 2, 1)
        flatten_1 = None
        image_positional_embeddings_3 = permute_252.unsqueeze(1)
        permute_252 = None
        query_64 = torch._C._nn.linear(
            point_embeddings,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        key_64 = torch._C._nn.linear(
            point_embeddings,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_64 = torch._C._nn.linear(
            point_embeddings,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        hidden_states_336 = query_64.reshape(1, 5, 8, 32)
        query_64 = None
        query_65 = hidden_states_336.transpose(1, 2)
        hidden_states_336 = None
        hidden_states_337 = key_64.reshape(1, 5, 8, 32)
        key_64 = None
        key_65 = hidden_states_337.transpose(1, 2)
        hidden_states_337 = None
        hidden_states_338 = value_64.reshape(1, 5, 8, 32)
        value_64 = None
        value_65 = hidden_states_338.transpose(1, 2)
        hidden_states_338 = None
        query_66 = query_65.contiguous()
        query_65 = None
        key_66 = key_65.contiguous()
        key_65 = None
        value_66 = value_65.contiguous()
        value_65 = None
        item_67 = (
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_scaling.item()
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_scaling = (
            None
        )
        attn_output_96 = torch._C._nn.scaled_dot_product_attention(
            query_66,
            key_66,
            value_66,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_67,
            is_causal=False,
        )
        query_66 = key_66 = value_66 = item_67 = None
        transpose_3 = attn_output_96.transpose(1, 2)
        attn_output_96 = None
        attn_output_97 = transpose_3.contiguous()
        transpose_3 = None
        attn_output_98 = attn_output_97.reshape(1, 1, 5, 256)
        attn_output_97 = None
        attn_output_99 = torch._C._nn.linear(
            attn_output_98,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_98 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        item_68 = (
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm1_eps.item()
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm1_eps = (
            None
        )
        queries = torch.nn.functional.layer_norm(
            attn_output_99,
            (256,),
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm1_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm1_parameters_bias_,
            item_68,
        )
        attn_output_99 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm1_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm1_parameters_bias_ = (item_68) = (
            None
        )
        query_67 = queries + point_embeddings
        key_67 = image_embeddings_2 + image_positional_embeddings_3
        query_68 = torch._C._nn.linear(
            query_67,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_q_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_q_proj_parameters_bias_,
        )
        query_67 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_q_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_q_proj_parameters_bias_ = (None)
        key_68 = torch._C._nn.linear(
            key_67,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_k_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_k_proj_parameters_bias_,
        )
        key_67 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_k_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_k_proj_parameters_bias_ = (None)
        value_67 = torch._C._nn.linear(
            image_embeddings_2,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_v_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_v_proj_parameters_bias_,
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_v_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_v_proj_parameters_bias_ = (None)
        hidden_states_339 = query_68.reshape(1, 5, 8, 16)
        query_68 = None
        query_69 = hidden_states_339.transpose(1, 2)
        hidden_states_339 = None
        hidden_states_340 = key_68.reshape(1, 4096, 8, 16)
        key_68 = None
        key_69 = hidden_states_340.transpose(1, 2)
        hidden_states_340 = None
        hidden_states_341 = value_67.reshape(1, 4096, 8, 16)
        value_67 = None
        value_68 = hidden_states_341.transpose(1, 2)
        hidden_states_341 = None
        query_70 = query_69.contiguous()
        query_69 = None
        key_70 = key_69.contiguous()
        key_69 = None
        value_69 = value_68.contiguous()
        value_68 = None
        item_69 = (
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_scaling.item()
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_scaling = (
            None
        )
        attn_output_100 = torch._C._nn.scaled_dot_product_attention(
            query_70,
            key_70,
            value_69,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_69,
            is_causal=False,
        )
        query_70 = key_70 = value_69 = item_69 = None
        transpose_7 = attn_output_100.transpose(1, 2)
        attn_output_100 = None
        attn_output_101 = transpose_7.contiguous()
        transpose_7 = None
        attn_output_102 = attn_output_101.reshape(1, 1, 5, 128)
        attn_output_101 = None
        attn_output_103 = torch._C._nn.linear(
            attn_output_102,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_out_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_out_proj_parameters_bias_,
        )
        attn_output_102 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_out_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_out_proj_parameters_bias_ = (None)
        queries_1 = queries + attn_output_103
        queries = attn_output_103 = None
        item_70 = (
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm2_eps.item()
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm2_eps = (
            None
        )
        queries_2 = torch.nn.functional.layer_norm(
            queries_1,
            (256,),
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm2_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm2_parameters_bias_,
            item_70,
        )
        queries_1 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm2_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm2_parameters_bias_ = (item_70) = (
            None
        )
        hidden_states_342 = torch._C._nn.linear(
            queries_2,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_mlp_modules_lin1_parameters_bias_,
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_343 = torch.nn.functional.relu(hidden_states_342, inplace=False)
        hidden_states_342 = None
        hidden_states_344 = torch._C._nn.linear(
            hidden_states_343,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_343 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_mlp_modules_lin2_parameters_bias_ = (None)
        queries_3 = queries_2 + hidden_states_344
        queries_2 = hidden_states_344 = None
        item_71 = (
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm3_eps.item()
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm3_eps = (
            None
        )
        queries_4 = torch.nn.functional.layer_norm(
            queries_3,
            (256,),
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm3_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm3_parameters_bias_,
            item_71,
        )
        queries_3 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm3_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm3_parameters_bias_ = (item_71) = (
            None
        )
        query_71 = queries_4 + point_embeddings
        key_71 = image_embeddings_2 + image_positional_embeddings_3
        query_72 = torch._C._nn.linear(
            key_71,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_q_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_q_proj_parameters_bias_,
        )
        key_71 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_q_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_q_proj_parameters_bias_ = (None)
        key_72 = torch._C._nn.linear(
            query_71,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_k_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_k_proj_parameters_bias_,
        )
        query_71 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_k_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_k_proj_parameters_bias_ = (None)
        value_70 = torch._C._nn.linear(
            queries_4,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_v_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_v_proj_parameters_bias_,
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_v_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_v_proj_parameters_bias_ = (None)
        hidden_states_345 = query_72.reshape(1, 4096, 8, 16)
        query_72 = None
        query_73 = hidden_states_345.transpose(1, 2)
        hidden_states_345 = None
        hidden_states_346 = key_72.reshape(1, 5, 8, 16)
        key_72 = None
        key_73 = hidden_states_346.transpose(1, 2)
        hidden_states_346 = None
        hidden_states_347 = value_70.reshape(1, 5, 8, 16)
        value_70 = None
        value_71 = hidden_states_347.transpose(1, 2)
        hidden_states_347 = None
        query_74 = query_73.contiguous()
        query_73 = None
        key_74 = key_73.contiguous()
        key_73 = None
        value_72 = value_71.contiguous()
        value_71 = None
        item_72 = (
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_scaling.item()
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_scaling = (
            None
        )
        attn_output_104 = torch._C._nn.scaled_dot_product_attention(
            query_74,
            key_74,
            value_72,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_72,
            is_causal=False,
        )
        query_74 = key_74 = value_72 = item_72 = None
        transpose_11 = attn_output_104.transpose(1, 2)
        attn_output_104 = None
        attn_output_105 = transpose_11.contiguous()
        transpose_11 = None
        attn_output_106 = attn_output_105.reshape(1, 1, 4096, 128)
        attn_output_105 = None
        attn_output_107 = torch._C._nn.linear(
            attn_output_106,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_out_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_out_proj_parameters_bias_,
        )
        attn_output_106 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_out_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_out_proj_parameters_bias_ = (None)
        keys = image_embeddings_2 + attn_output_107
        image_embeddings_2 = attn_output_107 = None
        item_73 = (
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm4_eps.item()
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm4_eps = (
            None
        )
        keys_1 = torch.nn.functional.layer_norm(
            keys,
            (256,),
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm4_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm4_parameters_bias_,
            item_73,
        )
        keys = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm4_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm4_parameters_bias_ = (item_73) = (
            None
        )
        query_75 = queries_4 + point_embeddings
        query_76 = torch._C._nn.linear(
            query_75,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        key_75 = torch._C._nn.linear(
            query_75,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        query_75 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_73 = torch._C._nn.linear(
            queries_4,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        hidden_states_348 = query_76.reshape(1, 5, 8, 32)
        query_76 = None
        query_77 = hidden_states_348.transpose(1, 2)
        hidden_states_348 = None
        hidden_states_349 = key_75.reshape(1, 5, 8, 32)
        key_75 = None
        key_76 = hidden_states_349.transpose(1, 2)
        hidden_states_349 = None
        hidden_states_350 = value_73.reshape(1, 5, 8, 32)
        value_73 = None
        value_74 = hidden_states_350.transpose(1, 2)
        hidden_states_350 = None
        query_78 = query_77.contiguous()
        query_77 = None
        key_77 = key_76.contiguous()
        key_76 = None
        value_75 = value_74.contiguous()
        value_74 = None
        item_74 = (
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_scaling.item()
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_scaling = (
            None
        )
        attn_output_108 = torch._C._nn.scaled_dot_product_attention(
            query_78,
            key_77,
            value_75,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_74,
            is_causal=False,
        )
        query_78 = key_77 = value_75 = item_74 = None
        transpose_15 = attn_output_108.transpose(1, 2)
        attn_output_108 = None
        attn_output_109 = transpose_15.contiguous()
        transpose_15 = None
        attn_output_110 = attn_output_109.reshape(1, 1, 5, 256)
        attn_output_109 = None
        attn_output_111 = torch._C._nn.linear(
            attn_output_110,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_110 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        queries_5 = queries_4 + attn_output_111
        queries_4 = attn_output_111 = None
        item_75 = (
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm1_eps.item()
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm1_eps = (
            None
        )
        queries_6 = torch.nn.functional.layer_norm(
            queries_5,
            (256,),
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm1_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm1_parameters_bias_,
            item_75,
        )
        queries_5 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm1_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm1_parameters_bias_ = (item_75) = (
            None
        )
        query_79 = queries_6 + point_embeddings
        key_78 = keys_1 + image_positional_embeddings_3
        query_80 = torch._C._nn.linear(
            query_79,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_q_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_q_proj_parameters_bias_,
        )
        query_79 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_q_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_q_proj_parameters_bias_ = (None)
        key_79 = torch._C._nn.linear(
            key_78,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_k_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_k_proj_parameters_bias_,
        )
        key_78 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_k_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_k_proj_parameters_bias_ = (None)
        value_76 = torch._C._nn.linear(
            keys_1,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_v_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_v_proj_parameters_bias_,
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_v_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_v_proj_parameters_bias_ = (None)
        hidden_states_351 = query_80.reshape(1, 5, 8, 16)
        query_80 = None
        query_81 = hidden_states_351.transpose(1, 2)
        hidden_states_351 = None
        hidden_states_352 = key_79.reshape(1, 4096, 8, 16)
        key_79 = None
        key_80 = hidden_states_352.transpose(1, 2)
        hidden_states_352 = None
        hidden_states_353 = value_76.reshape(1, 4096, 8, 16)
        value_76 = None
        value_77 = hidden_states_353.transpose(1, 2)
        hidden_states_353 = None
        query_82 = query_81.contiguous()
        query_81 = None
        key_81 = key_80.contiguous()
        key_80 = None
        value_78 = value_77.contiguous()
        value_77 = None
        item_76 = (
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_scaling.item()
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_scaling = (
            None
        )
        attn_output_112 = torch._C._nn.scaled_dot_product_attention(
            query_82,
            key_81,
            value_78,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_76,
            is_causal=False,
        )
        query_82 = key_81 = value_78 = item_76 = None
        transpose_19 = attn_output_112.transpose(1, 2)
        attn_output_112 = None
        attn_output_113 = transpose_19.contiguous()
        transpose_19 = None
        attn_output_114 = attn_output_113.reshape(1, 1, 5, 128)
        attn_output_113 = None
        attn_output_115 = torch._C._nn.linear(
            attn_output_114,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_out_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_out_proj_parameters_bias_,
        )
        attn_output_114 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_out_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_out_proj_parameters_bias_ = (None)
        queries_7 = queries_6 + attn_output_115
        queries_6 = attn_output_115 = None
        item_77 = (
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm2_eps.item()
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm2_eps = (
            None
        )
        queries_8 = torch.nn.functional.layer_norm(
            queries_7,
            (256,),
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm2_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm2_parameters_bias_,
            item_77,
        )
        queries_7 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm2_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm2_parameters_bias_ = (item_77) = (
            None
        )
        hidden_states_354 = torch._C._nn.linear(
            queries_8,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_mlp_modules_lin1_parameters_bias_,
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_355 = torch.nn.functional.relu(hidden_states_354, inplace=False)
        hidden_states_354 = None
        hidden_states_356 = torch._C._nn.linear(
            hidden_states_355,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_355 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_mlp_modules_lin2_parameters_bias_ = (None)
        queries_9 = queries_8 + hidden_states_356
        queries_8 = hidden_states_356 = None
        item_78 = (
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm3_eps.item()
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm3_eps = (
            None
        )
        queries_10 = torch.nn.functional.layer_norm(
            queries_9,
            (256,),
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm3_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm3_parameters_bias_,
            item_78,
        )
        queries_9 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm3_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm3_parameters_bias_ = (item_78) = (
            None
        )
        query_83 = queries_10 + point_embeddings
        key_82 = keys_1 + image_positional_embeddings_3
        query_84 = torch._C._nn.linear(
            key_82,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_q_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_q_proj_parameters_bias_,
        )
        key_82 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_q_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_q_proj_parameters_bias_ = (None)
        key_83 = torch._C._nn.linear(
            query_83,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_k_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_k_proj_parameters_bias_,
        )
        query_83 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_k_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_k_proj_parameters_bias_ = (None)
        value_79 = torch._C._nn.linear(
            queries_10,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_v_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_v_proj_parameters_bias_,
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_v_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_v_proj_parameters_bias_ = (None)
        hidden_states_357 = query_84.reshape(1, 4096, 8, 16)
        query_84 = None
        query_85 = hidden_states_357.transpose(1, 2)
        hidden_states_357 = None
        hidden_states_358 = key_83.reshape(1, 5, 8, 16)
        key_83 = None
        key_84 = hidden_states_358.transpose(1, 2)
        hidden_states_358 = None
        hidden_states_359 = value_79.reshape(1, 5, 8, 16)
        value_79 = None
        value_80 = hidden_states_359.transpose(1, 2)
        hidden_states_359 = None
        query_86 = query_85.contiguous()
        query_85 = None
        key_85 = key_84.contiguous()
        key_84 = None
        value_81 = value_80.contiguous()
        value_80 = None
        item_79 = (
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_scaling.item()
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_scaling = (
            None
        )
        attn_output_116 = torch._C._nn.scaled_dot_product_attention(
            query_86,
            key_85,
            value_81,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_79,
            is_causal=False,
        )
        query_86 = key_85 = value_81 = item_79 = None
        transpose_23 = attn_output_116.transpose(1, 2)
        attn_output_116 = None
        attn_output_117 = transpose_23.contiguous()
        transpose_23 = None
        attn_output_118 = attn_output_117.reshape(1, 1, 4096, 128)
        attn_output_117 = None
        attn_output_119 = torch._C._nn.linear(
            attn_output_118,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_out_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_out_proj_parameters_bias_,
        )
        attn_output_118 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_out_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_out_proj_parameters_bias_ = (None)
        keys_2 = keys_1 + attn_output_119
        keys_1 = attn_output_119 = None
        item_80 = (
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm4_eps.item()
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm4_eps = (
            None
        )
        keys_3 = torch.nn.functional.layer_norm(
            keys_2,
            (256,),
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm4_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm4_parameters_bias_,
            item_80,
        )
        keys_2 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm4_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm4_parameters_bias_ = (item_80) = (
            None
        )
        query_87 = queries_10 + point_embeddings
        point_embeddings = None
        key_86 = keys_3 + image_positional_embeddings_3
        image_positional_embeddings_3 = None
        query_88 = torch._C._nn.linear(
            query_87,
            l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_q_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_q_proj_parameters_bias_,
        )
        query_87 = l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_q_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_q_proj_parameters_bias_ = (None)
        key_87 = torch._C._nn.linear(
            key_86,
            l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_k_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_k_proj_parameters_bias_,
        )
        key_86 = l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_k_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_k_proj_parameters_bias_ = (None)
        value_82 = torch._C._nn.linear(
            keys_3,
            l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_v_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_v_proj_parameters_bias_,
        )
        l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_v_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_v_proj_parameters_bias_ = (None)
        hidden_states_360 = query_88.reshape(1, 5, 8, 16)
        query_88 = None
        query_89 = hidden_states_360.transpose(1, 2)
        hidden_states_360 = None
        hidden_states_361 = key_87.reshape(1, 4096, 8, 16)
        key_87 = None
        key_88 = hidden_states_361.transpose(1, 2)
        hidden_states_361 = None
        hidden_states_362 = value_82.reshape(1, 4096, 8, 16)
        value_82 = None
        value_83 = hidden_states_362.transpose(1, 2)
        hidden_states_362 = None
        query_90 = query_89.contiguous()
        query_89 = None
        key_89 = key_88.contiguous()
        key_88 = None
        value_84 = value_83.contiguous()
        value_83 = None
        item_81 = (
            l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_scaling.item()
        )
        l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_scaling = (
            None
        )
        attn_output_120 = torch._C._nn.scaled_dot_product_attention(
            query_90,
            key_89,
            value_84,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_81,
            is_causal=False,
        )
        query_90 = key_89 = value_84 = item_81 = None
        transpose_27 = attn_output_120.transpose(1, 2)
        attn_output_120 = None
        attn_output_121 = transpose_27.contiguous()
        transpose_27 = None
        attn_output_122 = attn_output_121.reshape(1, 1, 5, 128)
        attn_output_121 = None
        attn_output_123 = torch._C._nn.linear(
            attn_output_122,
            l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_out_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_out_proj_parameters_bias_,
        )
        attn_output_122 = l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_out_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_out_proj_parameters_bias_ = (None)
        queries_11 = queries_10 + attn_output_123
        queries_10 = attn_output_123 = None
        item_82 = (
            l_self_modules_mask_decoder_modules_transformer_modules_layer_norm_final_attn_eps.item()
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layer_norm_final_attn_eps = (
            None
        )
        queries_12 = torch.nn.functional.layer_norm(
            queries_11,
            (256,),
            l_self_modules_mask_decoder_modules_transformer_modules_layer_norm_final_attn_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layer_norm_final_attn_parameters_bias_,
            item_82,
        )
        queries_11 = l_self_modules_mask_decoder_modules_transformer_modules_layer_norm_final_attn_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layer_norm_final_attn_parameters_bias_ = (item_82) = (
            None
        )
        iou_token_out = queries_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                0,
                slice(None, None, None),
            )
        ]
        mask_tokens_out = queries_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, 5, None),
                slice(None, None, None),
            )
        ]
        queries_12 = None
        transpose_28 = keys_3.transpose(2, 3)
        keys_3 = None
        image_embeddings_3 = transpose_28.reshape(1, 256, 64, 64)
        transpose_28 = None
        upscaled_embedding = torch.conv_transpose2d(
            image_embeddings_3,
            l_self_modules_mask_decoder_modules_upscale_conv1_parameters_weight_,
            l_self_modules_mask_decoder_modules_upscale_conv1_parameters_bias_,
            (2, 2),
            (0, 0),
            (0, 0),
            1,
            (1, 1),
        )
        image_embeddings_3 = (
            l_self_modules_mask_decoder_modules_upscale_conv1_parameters_weight_
        ) = l_self_modules_mask_decoder_modules_upscale_conv1_parameters_bias_ = None
        x_8 = upscaled_embedding.float()
        upscaled_embedding = None
        u_2 = x_8.mean(1, keepdim=True)
        sub_71 = x_8 - u_2
        pow_3 = sub_71.pow(2)
        sub_71 = None
        s_2 = pow_3.mean(1, keepdim=True)
        pow_3 = None
        sub_72 = x_8 - u_2
        x_8 = u_2 = None
        item_83 = l_self_modules_mask_decoder_modules_upscale_layer_norm_eps.item()
        l_self_modules_mask_decoder_modules_upscale_layer_norm_eps = None
        add_185 = s_2 + item_83
        s_2 = item_83 = None
        sqrt_2 = torch.sqrt(add_185)
        add_185 = None
        x_9 = sub_72 / sqrt_2
        sub_72 = sqrt_2 = None
        x_10 = x_9.to(dtype=torch.float32)
        x_9 = None
        getitem_394 = (
            l_self_modules_mask_decoder_modules_upscale_layer_norm_parameters_weight_[
                (slice(None, None, None), None, None)
            ]
        )
        l_self_modules_mask_decoder_modules_upscale_layer_norm_parameters_weight_ = None
        mul_133 = getitem_394 * x_10
        getitem_394 = x_10 = None
        getitem_395 = (
            l_self_modules_mask_decoder_modules_upscale_layer_norm_parameters_bias_[
                (slice(None, None, None), None, None)
            ]
        )
        l_self_modules_mask_decoder_modules_upscale_layer_norm_parameters_bias_ = None
        x_11 = mul_133 + getitem_395
        mul_133 = getitem_395 = None
        upscaled_embedding_1 = torch._C._nn.gelu(x_11, approximate="none")
        x_11 = None
        conv_transpose2d_1 = torch.conv_transpose2d(
            upscaled_embedding_1,
            l_self_modules_mask_decoder_modules_upscale_conv2_parameters_weight_,
            l_self_modules_mask_decoder_modules_upscale_conv2_parameters_bias_,
            (2, 2),
            (0, 0),
            (0, 0),
            1,
            (1, 1),
        )
        upscaled_embedding_1 = (
            l_self_modules_mask_decoder_modules_upscale_conv2_parameters_weight_
        ) = l_self_modules_mask_decoder_modules_upscale_conv2_parameters_bias_ = None
        upscaled_embedding_2 = torch._C._nn.gelu(conv_transpose2d_1, approximate="none")
        conv_transpose2d_1 = None
        getitem_396 = mask_tokens_out[
            (
                slice(None, None, None),
                slice(None, None, None),
                0,
                slice(None, None, None),
            )
        ]
        hidden_states_363 = torch._C._nn.linear(
            getitem_396,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_proj_in_parameters_weight_,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_proj_in_parameters_bias_,
        )
        getitem_396 = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_proj_in_parameters_weight_ = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_proj_in_parameters_bias_ = (None)
        hidden_states_364 = torch.nn.functional.relu(hidden_states_363, inplace=False)
        hidden_states_363 = None
        linear_161 = torch._C._nn.linear(
            hidden_states_364,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_layers_modules_0_parameters_weight_,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_layers_modules_0_parameters_bias_,
        )
        hidden_states_364 = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_layers_modules_0_parameters_weight_ = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_layers_modules_0_parameters_bias_ = (None)
        hidden_states_365 = torch.nn.functional.relu(linear_161, inplace=False)
        linear_161 = None
        hidden_states_366 = torch._C._nn.linear(
            hidden_states_365,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_proj_out_parameters_weight_,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_proj_out_parameters_bias_,
        )
        hidden_states_365 = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_proj_out_parameters_weight_ = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_proj_out_parameters_bias_ = (None)
        getitem_397 = mask_tokens_out[
            (
                slice(None, None, None),
                slice(None, None, None),
                1,
                slice(None, None, None),
            )
        ]
        hidden_states_367 = torch._C._nn.linear(
            getitem_397,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_proj_in_parameters_weight_,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_proj_in_parameters_bias_,
        )
        getitem_397 = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_proj_in_parameters_weight_ = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_proj_in_parameters_bias_ = (None)
        hidden_states_368 = torch.nn.functional.relu(hidden_states_367, inplace=False)
        hidden_states_367 = None
        linear_164 = torch._C._nn.linear(
            hidden_states_368,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_layers_modules_0_parameters_weight_,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_layers_modules_0_parameters_bias_,
        )
        hidden_states_368 = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_layers_modules_0_parameters_weight_ = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_layers_modules_0_parameters_bias_ = (None)
        hidden_states_369 = torch.nn.functional.relu(linear_164, inplace=False)
        linear_164 = None
        hidden_states_370 = torch._C._nn.linear(
            hidden_states_369,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_proj_out_parameters_weight_,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_proj_out_parameters_bias_,
        )
        hidden_states_369 = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_proj_out_parameters_weight_ = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_proj_out_parameters_bias_ = (None)
        getitem_398 = mask_tokens_out[
            (
                slice(None, None, None),
                slice(None, None, None),
                2,
                slice(None, None, None),
            )
        ]
        hidden_states_371 = torch._C._nn.linear(
            getitem_398,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_proj_in_parameters_weight_,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_proj_in_parameters_bias_,
        )
        getitem_398 = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_proj_in_parameters_weight_ = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_proj_in_parameters_bias_ = (None)
        hidden_states_372 = torch.nn.functional.relu(hidden_states_371, inplace=False)
        hidden_states_371 = None
        linear_167 = torch._C._nn.linear(
            hidden_states_372,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_layers_modules_0_parameters_weight_,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_layers_modules_0_parameters_bias_,
        )
        hidden_states_372 = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_layers_modules_0_parameters_weight_ = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_layers_modules_0_parameters_bias_ = (None)
        hidden_states_373 = torch.nn.functional.relu(linear_167, inplace=False)
        linear_167 = None
        hidden_states_374 = torch._C._nn.linear(
            hidden_states_373,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_proj_out_parameters_weight_,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_proj_out_parameters_bias_,
        )
        hidden_states_373 = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_proj_out_parameters_weight_ = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_proj_out_parameters_bias_ = (None)
        getitem_399 = mask_tokens_out[
            (
                slice(None, None, None),
                slice(None, None, None),
                3,
                slice(None, None, None),
            )
        ]
        mask_tokens_out = None
        hidden_states_375 = torch._C._nn.linear(
            getitem_399,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_proj_in_parameters_weight_,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_proj_in_parameters_bias_,
        )
        getitem_399 = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_proj_in_parameters_weight_ = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_proj_in_parameters_bias_ = (None)
        hidden_states_376 = torch.nn.functional.relu(hidden_states_375, inplace=False)
        hidden_states_375 = None
        linear_170 = torch._C._nn.linear(
            hidden_states_376,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_layers_modules_0_parameters_weight_,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_layers_modules_0_parameters_bias_,
        )
        hidden_states_376 = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_layers_modules_0_parameters_weight_ = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_layers_modules_0_parameters_bias_ = (None)
        hidden_states_377 = torch.nn.functional.relu(linear_170, inplace=False)
        linear_170 = None
        hidden_states_378 = torch._C._nn.linear(
            hidden_states_377,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_proj_out_parameters_weight_,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_proj_out_parameters_bias_,
        )
        hidden_states_377 = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_proj_out_parameters_weight_ = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_proj_out_parameters_bias_ = (None)
        hyper_in = torch.stack(
            [
                hidden_states_366,
                hidden_states_370,
                hidden_states_374,
                hidden_states_378,
            ],
            dim=2,
        )
        hidden_states_366 = (
            hidden_states_370
        ) = hidden_states_374 = hidden_states_378 = None
        upscaled_embedding_3 = upscaled_embedding_2.reshape(1, 1, 32, 65536)
        upscaled_embedding_2 = None
        matmul_1 = hyper_in @ upscaled_embedding_3
        hyper_in = upscaled_embedding_3 = None
        masks = matmul_1.reshape(1, 1, -1, 256, 256)
        matmul_1 = None
        hidden_states_379 = torch._C._nn.linear(
            iou_token_out,
            l_self_modules_mask_decoder_modules_iou_prediction_head_modules_proj_in_parameters_weight_,
            l_self_modules_mask_decoder_modules_iou_prediction_head_modules_proj_in_parameters_bias_,
        )
        iou_token_out = l_self_modules_mask_decoder_modules_iou_prediction_head_modules_proj_in_parameters_weight_ = l_self_modules_mask_decoder_modules_iou_prediction_head_modules_proj_in_parameters_bias_ = (None)
        hidden_states_380 = torch.nn.functional.relu(hidden_states_379, inplace=False)
        hidden_states_379 = None
        linear_173 = torch._C._nn.linear(
            hidden_states_380,
            l_self_modules_mask_decoder_modules_iou_prediction_head_modules_layers_modules_0_parameters_weight_,
            l_self_modules_mask_decoder_modules_iou_prediction_head_modules_layers_modules_0_parameters_bias_,
        )
        hidden_states_380 = l_self_modules_mask_decoder_modules_iou_prediction_head_modules_layers_modules_0_parameters_weight_ = l_self_modules_mask_decoder_modules_iou_prediction_head_modules_layers_modules_0_parameters_bias_ = (None)
        hidden_states_381 = torch.nn.functional.relu(linear_173, inplace=False)
        linear_173 = None
        hidden_states_382 = torch._C._nn.linear(
            hidden_states_381,
            l_self_modules_mask_decoder_modules_iou_prediction_head_modules_proj_out_parameters_weight_,
            l_self_modules_mask_decoder_modules_iou_prediction_head_modules_proj_out_parameters_bias_,
        )
        hidden_states_381 = l_self_modules_mask_decoder_modules_iou_prediction_head_modules_proj_out_parameters_weight_ = l_self_modules_mask_decoder_modules_iou_prediction_head_modules_proj_out_parameters_bias_ = (None)
        masks_1 = masks[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        masks = None
        iou_pred = hidden_states_382[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        hidden_states_382 = None
        return (iou_pred, masks_1)
