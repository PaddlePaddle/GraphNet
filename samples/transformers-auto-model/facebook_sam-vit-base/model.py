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
            (768,),
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
        hidden_states_3 = hidden_states_2.reshape(1, 5, 14, 5, 14, 768)
        hidden_states_2 = None
        permute_2 = hidden_states_3.permute(0, 1, 3, 2, 4, 5)
        hidden_states_3 = None
        contiguous = permute_2.contiguous()
        permute_2 = None
        windows = contiguous.reshape(-1, 14, 14, 768)
        contiguous = None
        linear = torch._C._nn.linear(
            windows,
            l_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        windows = l_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_2 = linear.reshape(25, 196, 3, 12, -1)
        linear = None
        qkv = reshape_2.permute(2, 0, 3, 1, 4)
        reshape_2 = None
        reshape_3 = qkv.reshape(3, 300, 196, -1)
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
        reshaped_query = query.reshape(300, 14, 14, 64)
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
        decomposed_rel_pos_1 = decomposed_rel_pos.reshape(25, 12, 196, 196)
        decomposed_rel_pos = None
        query_1 = query.view(25, 12, 196, -1)
        query = None
        key_1 = key.view(25, 12, 196, -1)
        key = None
        value_1 = value.view(25, 12, 196, -1)
        value = None
        attn_output = torch._C._nn.scaled_dot_product_attention(
            query_1, key_1, value_1, attn_mask=decomposed_rel_pos_1
        )
        query_1 = key_1 = value_1 = decomposed_rel_pos_1 = None
        view_3 = attn_output.view(25, 12, 14, 14, -1)
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
            (768,),
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
            (768,),
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
        hidden_states_14 = hidden_states_13.reshape(1, 5, 14, 5, 14, 768)
        hidden_states_13 = None
        permute_10 = hidden_states_14.permute(0, 1, 3, 2, 4, 5)
        hidden_states_14 = None
        contiguous_3 = permute_10.contiguous()
        permute_10 = None
        windows_1 = contiguous_3.reshape(-1, 14, 14, 768)
        contiguous_3 = None
        linear_4 = torch._C._nn.linear(
            windows_1,
            l_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_modules_qkv_parameters_bias_,
        )
        windows_1 = l_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_15 = linear_4.reshape(25, 196, 3, 12, -1)
        linear_4 = None
        qkv_1 = reshape_15.permute(2, 0, 3, 1, 4)
        reshape_15 = None
        reshape_16 = qkv_1.reshape(3, 300, 196, -1)
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
        reshaped_query_1 = query_2.reshape(300, 14, 14, 64)
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
        decomposed_rel_pos_3 = decomposed_rel_pos_2.reshape(25, 12, 196, 196)
        decomposed_rel_pos_2 = None
        query_3 = query_2.view(25, 12, 196, -1)
        query_2 = None
        key_3 = key_2.view(25, 12, 196, -1)
        key_2 = None
        value_3 = value_2.view(25, 12, 196, -1)
        value_2 = None
        attn_output_3 = torch._C._nn.scaled_dot_product_attention(
            query_3, key_3, value_3, attn_mask=decomposed_rel_pos_3
        )
        query_3 = key_3 = value_3 = decomposed_rel_pos_3 = None
        view_7 = attn_output_3.view(25, 12, 14, 14, -1)
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
            (768,),
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
            (768,),
            l_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_bias_,
            item_5,
        )
        l_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_bias_ = (item_5) = (
            None
        )
        linear_8 = torch._C._nn.linear(
            hidden_states_23,
            l_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_modules_qkv_parameters_bias_,
        )
        hidden_states_23 = l_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_26 = linear_8.reshape(1, 4096, 3, 12, -1)
        linear_8 = None
        qkv_2 = reshape_26.permute(2, 0, 3, 1, 4)
        reshape_26 = None
        reshape_27 = qkv_2.reshape(3, 12, 4096, -1)
        qkv_2 = None
        unbind_2 = reshape_27.unbind(0)
        reshape_27 = None
        query_4 = unbind_2[0]
        key_4 = unbind_2[1]
        value_4 = unbind_2[2]
        unbind_2 = None
        reshape_28 = l_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_parameters_rel_pos_h_.reshape(
            1, 127, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_19 = reshape_28.permute(0, 2, 1)
        reshape_28 = None
        rel_pos_resized_8 = torch.nn.functional.interpolate(
            permute_19, size=127, mode="linear"
        )
        permute_19 = None
        reshape_29 = rel_pos_resized_8.reshape(-1, 127)
        rel_pos_resized_8 = None
        rel_pos_resized_9 = reshape_29.permute(1, 0)
        reshape_29 = None
        arange_8 = torch.arange(64)
        getitem_35 = arange_8[(slice(None, None, None), None)]
        arange_8 = None
        q_coords_4 = getitem_35 * 1.0
        getitem_35 = None
        arange_9 = torch.arange(64)
        getitem_36 = arange_9[(None, slice(None, None, None))]
        arange_9 = None
        k_coords_4 = getitem_36 * 1.0
        getitem_36 = None
        sub_7 = q_coords_4 - k_coords_4
        q_coords_4 = k_coords_4 = None
        relative_coords_4 = sub_7 + 63.0
        sub_7 = None
        long_4 = relative_coords_4.long()
        relative_coords_4 = None
        relative_position_height_2 = rel_pos_resized_9[long_4]
        rel_pos_resized_9 = long_4 = None
        reshape_30 = l_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_parameters_rel_pos_w_.reshape(
            1, 127, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_21 = reshape_30.permute(0, 2, 1)
        reshape_30 = None
        rel_pos_resized_10 = torch.nn.functional.interpolate(
            permute_21, size=127, mode="linear"
        )
        permute_21 = None
        reshape_31 = rel_pos_resized_10.reshape(-1, 127)
        rel_pos_resized_10 = None
        rel_pos_resized_11 = reshape_31.permute(1, 0)
        reshape_31 = None
        arange_10 = torch.arange(64)
        getitem_38 = arange_10[(slice(None, None, None), None)]
        arange_10 = None
        q_coords_5 = getitem_38 * 1.0
        getitem_38 = None
        arange_11 = torch.arange(64)
        getitem_39 = arange_11[(None, slice(None, None, None))]
        arange_11 = None
        k_coords_5 = getitem_39 * 1.0
        getitem_39 = None
        sub_8 = q_coords_5 - k_coords_5
        q_coords_5 = k_coords_5 = None
        relative_coords_5 = sub_8 + 63.0
        sub_8 = None
        long_5 = relative_coords_5.long()
        relative_coords_5 = None
        relative_position_width_2 = rel_pos_resized_11[long_5]
        rel_pos_resized_11 = long_5 = None
        reshaped_query_2 = query_4.reshape(12, 64, 64, 64)
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
        decomposed_rel_pos_5 = decomposed_rel_pos_4.reshape(1, 12, 4096, 4096)
        decomposed_rel_pos_4 = None
        query_5 = query_4.view(1, 12, 4096, -1)
        query_4 = None
        key_5 = key_4.view(1, 12, 4096, -1)
        key_4 = None
        value_5 = value_4.view(1, 12, 4096, -1)
        value_4 = None
        attn_output_6 = torch._C._nn.scaled_dot_product_attention(
            query_5, key_5, value_5, attn_mask=decomposed_rel_pos_5
        )
        query_5 = key_5 = value_5 = decomposed_rel_pos_5 = None
        view_11 = attn_output_6.view(1, 12, 64, 64, -1)
        attn_output_6 = None
        permute_23 = view_11.permute(0, 2, 3, 1, 4)
        view_11 = None
        attn_output_7 = permute_23.reshape(1, 64, 64, -1)
        permute_23 = None
        attn_output_8 = torch._C._nn.linear(
            attn_output_7,
            l_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_7 = l_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_24 = hidden_states_22 + attn_output_8
        hidden_states_22 = attn_output_8 = None
        item_6 = (
            l_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_2 = torch.nn.functional.layer_norm(
            hidden_states_24,
            (768,),
            l_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_bias_,
            item_6,
        )
        l_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_bias_ = (item_6) = (
            None
        )
        hidden_states_25 = torch._C._nn.linear(
            layernorm_output_2,
            l_self_modules_vision_encoder_modules_layers_modules_2_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_2_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_2 = l_self_modules_vision_encoder_modules_layers_modules_2_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_2_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_26 = torch._C._nn.gelu(hidden_states_25)
        hidden_states_25 = None
        hidden_states_27 = torch._C._nn.linear(
            hidden_states_26,
            l_self_modules_vision_encoder_modules_layers_modules_2_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_2_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_26 = l_self_modules_vision_encoder_modules_layers_modules_2_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_2_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_28 = hidden_states_24 + hidden_states_27
        hidden_states_24 = hidden_states_27 = None
        item_7 = (
            l_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm1_eps = (
            None
        )
        hidden_states_29 = torch.nn.functional.layer_norm(
            hidden_states_28,
            (768,),
            l_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_bias_,
            item_7,
        )
        l_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_bias_ = (item_7) = (
            None
        )
        hidden_states_30 = torch._C._nn.pad(
            hidden_states_29, (0, 0, 0, 6, 0, 6), "constant", None
        )
        hidden_states_29 = None
        hidden_states_31 = hidden_states_30.reshape(1, 5, 14, 5, 14, 768)
        hidden_states_30 = None
        permute_24 = hidden_states_31.permute(0, 1, 3, 2, 4, 5)
        hidden_states_31 = None
        contiguous_6 = permute_24.contiguous()
        permute_24 = None
        windows_2 = contiguous_6.reshape(-1, 14, 14, 768)
        contiguous_6 = None
        linear_12 = torch._C._nn.linear(
            windows_2,
            l_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_modules_qkv_parameters_bias_,
        )
        windows_2 = l_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_37 = linear_12.reshape(25, 196, 3, 12, -1)
        linear_12 = None
        qkv_3 = reshape_37.permute(2, 0, 3, 1, 4)
        reshape_37 = None
        reshape_38 = qkv_3.reshape(3, 300, 196, -1)
        qkv_3 = None
        unbind_3 = reshape_38.unbind(0)
        reshape_38 = None
        query_6 = unbind_3[0]
        key_6 = unbind_3[1]
        value_6 = unbind_3[2]
        unbind_3 = None
        reshape_39 = l_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_parameters_rel_pos_h_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_26 = reshape_39.permute(0, 2, 1)
        reshape_39 = None
        rel_pos_resized_12 = torch.nn.functional.interpolate(
            permute_26, size=27, mode="linear"
        )
        permute_26 = None
        reshape_40 = rel_pos_resized_12.reshape(-1, 27)
        rel_pos_resized_12 = None
        rel_pos_resized_13 = reshape_40.permute(1, 0)
        reshape_40 = None
        arange_12 = torch.arange(14)
        getitem_46 = arange_12[(slice(None, None, None), None)]
        arange_12 = None
        q_coords_6 = getitem_46 * 1.0
        getitem_46 = None
        arange_13 = torch.arange(14)
        getitem_47 = arange_13[(None, slice(None, None, None))]
        arange_13 = None
        k_coords_6 = getitem_47 * 1.0
        getitem_47 = None
        sub_9 = q_coords_6 - k_coords_6
        q_coords_6 = k_coords_6 = None
        relative_coords_6 = sub_9 + 13.0
        sub_9 = None
        long_6 = relative_coords_6.long()
        relative_coords_6 = None
        relative_position_height_3 = rel_pos_resized_13[long_6]
        rel_pos_resized_13 = long_6 = None
        reshape_41 = l_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_parameters_rel_pos_w_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_28 = reshape_41.permute(0, 2, 1)
        reshape_41 = None
        rel_pos_resized_14 = torch.nn.functional.interpolate(
            permute_28, size=27, mode="linear"
        )
        permute_28 = None
        reshape_42 = rel_pos_resized_14.reshape(-1, 27)
        rel_pos_resized_14 = None
        rel_pos_resized_15 = reshape_42.permute(1, 0)
        reshape_42 = None
        arange_14 = torch.arange(14)
        getitem_49 = arange_14[(slice(None, None, None), None)]
        arange_14 = None
        q_coords_7 = getitem_49 * 1.0
        getitem_49 = None
        arange_15 = torch.arange(14)
        getitem_50 = arange_15[(None, slice(None, None, None))]
        arange_15 = None
        k_coords_7 = getitem_50 * 1.0
        getitem_50 = None
        sub_10 = q_coords_7 - k_coords_7
        q_coords_7 = k_coords_7 = None
        relative_coords_7 = sub_10 + 13.0
        sub_10 = None
        long_7 = relative_coords_7.long()
        relative_coords_7 = None
        relative_position_width_3 = rel_pos_resized_15[long_7]
        rel_pos_resized_15 = long_7 = None
        reshaped_query_3 = query_6.reshape(300, 14, 14, 64)
        rel_h_3 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_3, relative_position_height_3
        )
        relative_position_height_3 = None
        rel_w_3 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_3, relative_position_width_3
        )
        reshaped_query_3 = relative_position_width_3 = None
        getitem_52 = rel_h_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_3 = None
        getitem_53 = rel_w_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_3 = None
        decomposed_rel_pos_6 = getitem_52 + getitem_53
        getitem_52 = getitem_53 = None
        decomposed_rel_pos_7 = decomposed_rel_pos_6.reshape(25, 12, 196, 196)
        decomposed_rel_pos_6 = None
        query_7 = query_6.view(25, 12, 196, -1)
        query_6 = None
        key_7 = key_6.view(25, 12, 196, -1)
        key_6 = None
        value_7 = value_6.view(25, 12, 196, -1)
        value_6 = None
        attn_output_9 = torch._C._nn.scaled_dot_product_attention(
            query_7, key_7, value_7, attn_mask=decomposed_rel_pos_7
        )
        query_7 = key_7 = value_7 = decomposed_rel_pos_7 = None
        view_15 = attn_output_9.view(25, 12, 14, 14, -1)
        attn_output_9 = None
        permute_30 = view_15.permute(0, 2, 3, 1, 4)
        view_15 = None
        attn_output_10 = permute_30.reshape(25, 14, 14, -1)
        permute_30 = None
        attn_output_11 = torch._C._nn.linear(
            attn_output_10,
            l_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_10 = l_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_32 = attn_output_11.reshape(1, 5, 5, 14, 14, -1)
        attn_output_11 = None
        permute_31 = hidden_states_32.permute(0, 1, 3, 2, 4, 5)
        hidden_states_32 = None
        contiguous_7 = permute_31.contiguous()
        permute_31 = None
        hidden_states_33 = contiguous_7.reshape(1, 70, 70, -1)
        contiguous_7 = None
        getitem_54 = hidden_states_33[
            (
                slice(None, None, None),
                slice(None, 64, None),
                slice(None, 64, None),
                slice(None, None, None),
            )
        ]
        hidden_states_33 = None
        hidden_states_34 = getitem_54.contiguous()
        getitem_54 = None
        hidden_states_35 = hidden_states_28 + hidden_states_34
        hidden_states_28 = hidden_states_34 = None
        item_8 = (
            l_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_3 = torch.nn.functional.layer_norm(
            hidden_states_35,
            (768,),
            l_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_bias_,
            item_8,
        )
        l_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_bias_ = (item_8) = (
            None
        )
        hidden_states_36 = torch._C._nn.linear(
            layernorm_output_3,
            l_self_modules_vision_encoder_modules_layers_modules_3_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_3_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_3 = l_self_modules_vision_encoder_modules_layers_modules_3_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_3_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_37 = torch._C._nn.gelu(hidden_states_36)
        hidden_states_36 = None
        hidden_states_38 = torch._C._nn.linear(
            hidden_states_37,
            l_self_modules_vision_encoder_modules_layers_modules_3_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_3_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_37 = l_self_modules_vision_encoder_modules_layers_modules_3_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_3_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_39 = hidden_states_35 + hidden_states_38
        hidden_states_35 = hidden_states_38 = None
        item_9 = (
            l_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm1_eps = (
            None
        )
        hidden_states_40 = torch.nn.functional.layer_norm(
            hidden_states_39,
            (768,),
            l_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_bias_,
            item_9,
        )
        l_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_bias_ = (item_9) = (
            None
        )
        hidden_states_41 = torch._C._nn.pad(
            hidden_states_40, (0, 0, 0, 6, 0, 6), "constant", None
        )
        hidden_states_40 = None
        hidden_states_42 = hidden_states_41.reshape(1, 5, 14, 5, 14, 768)
        hidden_states_41 = None
        permute_32 = hidden_states_42.permute(0, 1, 3, 2, 4, 5)
        hidden_states_42 = None
        contiguous_9 = permute_32.contiguous()
        permute_32 = None
        windows_3 = contiguous_9.reshape(-1, 14, 14, 768)
        contiguous_9 = None
        linear_16 = torch._C._nn.linear(
            windows_3,
            l_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_modules_qkv_parameters_bias_,
        )
        windows_3 = l_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_50 = linear_16.reshape(25, 196, 3, 12, -1)
        linear_16 = None
        qkv_4 = reshape_50.permute(2, 0, 3, 1, 4)
        reshape_50 = None
        reshape_51 = qkv_4.reshape(3, 300, 196, -1)
        qkv_4 = None
        unbind_4 = reshape_51.unbind(0)
        reshape_51 = None
        query_8 = unbind_4[0]
        key_8 = unbind_4[1]
        value_8 = unbind_4[2]
        unbind_4 = None
        reshape_52 = l_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_parameters_rel_pos_h_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_34 = reshape_52.permute(0, 2, 1)
        reshape_52 = None
        rel_pos_resized_16 = torch.nn.functional.interpolate(
            permute_34, size=27, mode="linear"
        )
        permute_34 = None
        reshape_53 = rel_pos_resized_16.reshape(-1, 27)
        rel_pos_resized_16 = None
        rel_pos_resized_17 = reshape_53.permute(1, 0)
        reshape_53 = None
        arange_16 = torch.arange(14)
        getitem_58 = arange_16[(slice(None, None, None), None)]
        arange_16 = None
        q_coords_8 = getitem_58 * 1.0
        getitem_58 = None
        arange_17 = torch.arange(14)
        getitem_59 = arange_17[(None, slice(None, None, None))]
        arange_17 = None
        k_coords_8 = getitem_59 * 1.0
        getitem_59 = None
        sub_11 = q_coords_8 - k_coords_8
        q_coords_8 = k_coords_8 = None
        relative_coords_8 = sub_11 + 13.0
        sub_11 = None
        long_8 = relative_coords_8.long()
        relative_coords_8 = None
        relative_position_height_4 = rel_pos_resized_17[long_8]
        rel_pos_resized_17 = long_8 = None
        reshape_54 = l_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_parameters_rel_pos_w_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_36 = reshape_54.permute(0, 2, 1)
        reshape_54 = None
        rel_pos_resized_18 = torch.nn.functional.interpolate(
            permute_36, size=27, mode="linear"
        )
        permute_36 = None
        reshape_55 = rel_pos_resized_18.reshape(-1, 27)
        rel_pos_resized_18 = None
        rel_pos_resized_19 = reshape_55.permute(1, 0)
        reshape_55 = None
        arange_18 = torch.arange(14)
        getitem_61 = arange_18[(slice(None, None, None), None)]
        arange_18 = None
        q_coords_9 = getitem_61 * 1.0
        getitem_61 = None
        arange_19 = torch.arange(14)
        getitem_62 = arange_19[(None, slice(None, None, None))]
        arange_19 = None
        k_coords_9 = getitem_62 * 1.0
        getitem_62 = None
        sub_12 = q_coords_9 - k_coords_9
        q_coords_9 = k_coords_9 = None
        relative_coords_9 = sub_12 + 13.0
        sub_12 = None
        long_9 = relative_coords_9.long()
        relative_coords_9 = None
        relative_position_width_4 = rel_pos_resized_19[long_9]
        rel_pos_resized_19 = long_9 = None
        reshaped_query_4 = query_8.reshape(300, 14, 14, 64)
        rel_h_4 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_4, relative_position_height_4
        )
        relative_position_height_4 = None
        rel_w_4 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_4, relative_position_width_4
        )
        reshaped_query_4 = relative_position_width_4 = None
        getitem_64 = rel_h_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_4 = None
        getitem_65 = rel_w_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_4 = None
        decomposed_rel_pos_8 = getitem_64 + getitem_65
        getitem_64 = getitem_65 = None
        decomposed_rel_pos_9 = decomposed_rel_pos_8.reshape(25, 12, 196, 196)
        decomposed_rel_pos_8 = None
        query_9 = query_8.view(25, 12, 196, -1)
        query_8 = None
        key_9 = key_8.view(25, 12, 196, -1)
        key_8 = None
        value_9 = value_8.view(25, 12, 196, -1)
        value_8 = None
        attn_output_12 = torch._C._nn.scaled_dot_product_attention(
            query_9, key_9, value_9, attn_mask=decomposed_rel_pos_9
        )
        query_9 = key_9 = value_9 = decomposed_rel_pos_9 = None
        view_19 = attn_output_12.view(25, 12, 14, 14, -1)
        attn_output_12 = None
        permute_38 = view_19.permute(0, 2, 3, 1, 4)
        view_19 = None
        attn_output_13 = permute_38.reshape(25, 14, 14, -1)
        permute_38 = None
        attn_output_14 = torch._C._nn.linear(
            attn_output_13,
            l_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_13 = l_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_43 = attn_output_14.reshape(1, 5, 5, 14, 14, -1)
        attn_output_14 = None
        permute_39 = hidden_states_43.permute(0, 1, 3, 2, 4, 5)
        hidden_states_43 = None
        contiguous_10 = permute_39.contiguous()
        permute_39 = None
        hidden_states_44 = contiguous_10.reshape(1, 70, 70, -1)
        contiguous_10 = None
        getitem_66 = hidden_states_44[
            (
                slice(None, None, None),
                slice(None, 64, None),
                slice(None, 64, None),
                slice(None, None, None),
            )
        ]
        hidden_states_44 = None
        hidden_states_45 = getitem_66.contiguous()
        getitem_66 = None
        hidden_states_46 = hidden_states_39 + hidden_states_45
        hidden_states_39 = hidden_states_45 = None
        item_10 = (
            l_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_4 = torch.nn.functional.layer_norm(
            hidden_states_46,
            (768,),
            l_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_bias_,
            item_10,
        )
        l_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_bias_ = (item_10) = (
            None
        )
        hidden_states_47 = torch._C._nn.linear(
            layernorm_output_4,
            l_self_modules_vision_encoder_modules_layers_modules_4_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_4_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_4 = l_self_modules_vision_encoder_modules_layers_modules_4_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_4_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_48 = torch._C._nn.gelu(hidden_states_47)
        hidden_states_47 = None
        hidden_states_49 = torch._C._nn.linear(
            hidden_states_48,
            l_self_modules_vision_encoder_modules_layers_modules_4_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_4_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_48 = l_self_modules_vision_encoder_modules_layers_modules_4_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_4_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_50 = hidden_states_46 + hidden_states_49
        hidden_states_46 = hidden_states_49 = None
        item_11 = (
            l_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm1_eps = (
            None
        )
        hidden_states_51 = torch.nn.functional.layer_norm(
            hidden_states_50,
            (768,),
            l_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_bias_,
            item_11,
        )
        l_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_bias_ = (item_11) = (
            None
        )
        linear_20 = torch._C._nn.linear(
            hidden_states_51,
            l_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_modules_qkv_parameters_bias_,
        )
        hidden_states_51 = l_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_61 = linear_20.reshape(1, 4096, 3, 12, -1)
        linear_20 = None
        qkv_5 = reshape_61.permute(2, 0, 3, 1, 4)
        reshape_61 = None
        reshape_62 = qkv_5.reshape(3, 12, 4096, -1)
        qkv_5 = None
        unbind_5 = reshape_62.unbind(0)
        reshape_62 = None
        query_10 = unbind_5[0]
        key_10 = unbind_5[1]
        value_10 = unbind_5[2]
        unbind_5 = None
        reshape_63 = l_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_parameters_rel_pos_h_.reshape(
            1, 127, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_41 = reshape_63.permute(0, 2, 1)
        reshape_63 = None
        rel_pos_resized_20 = torch.nn.functional.interpolate(
            permute_41, size=127, mode="linear"
        )
        permute_41 = None
        reshape_64 = rel_pos_resized_20.reshape(-1, 127)
        rel_pos_resized_20 = None
        rel_pos_resized_21 = reshape_64.permute(1, 0)
        reshape_64 = None
        arange_20 = torch.arange(64)
        getitem_70 = arange_20[(slice(None, None, None), None)]
        arange_20 = None
        q_coords_10 = getitem_70 * 1.0
        getitem_70 = None
        arange_21 = torch.arange(64)
        getitem_71 = arange_21[(None, slice(None, None, None))]
        arange_21 = None
        k_coords_10 = getitem_71 * 1.0
        getitem_71 = None
        sub_13 = q_coords_10 - k_coords_10
        q_coords_10 = k_coords_10 = None
        relative_coords_10 = sub_13 + 63.0
        sub_13 = None
        long_10 = relative_coords_10.long()
        relative_coords_10 = None
        relative_position_height_5 = rel_pos_resized_21[long_10]
        rel_pos_resized_21 = long_10 = None
        reshape_65 = l_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_parameters_rel_pos_w_.reshape(
            1, 127, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_43 = reshape_65.permute(0, 2, 1)
        reshape_65 = None
        rel_pos_resized_22 = torch.nn.functional.interpolate(
            permute_43, size=127, mode="linear"
        )
        permute_43 = None
        reshape_66 = rel_pos_resized_22.reshape(-1, 127)
        rel_pos_resized_22 = None
        rel_pos_resized_23 = reshape_66.permute(1, 0)
        reshape_66 = None
        arange_22 = torch.arange(64)
        getitem_73 = arange_22[(slice(None, None, None), None)]
        arange_22 = None
        q_coords_11 = getitem_73 * 1.0
        getitem_73 = None
        arange_23 = torch.arange(64)
        getitem_74 = arange_23[(None, slice(None, None, None))]
        arange_23 = None
        k_coords_11 = getitem_74 * 1.0
        getitem_74 = None
        sub_14 = q_coords_11 - k_coords_11
        q_coords_11 = k_coords_11 = None
        relative_coords_11 = sub_14 + 63.0
        sub_14 = None
        long_11 = relative_coords_11.long()
        relative_coords_11 = None
        relative_position_width_5 = rel_pos_resized_23[long_11]
        rel_pos_resized_23 = long_11 = None
        reshaped_query_5 = query_10.reshape(12, 64, 64, 64)
        rel_h_5 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_5, relative_position_height_5
        )
        relative_position_height_5 = None
        rel_w_5 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_5, relative_position_width_5
        )
        reshaped_query_5 = relative_position_width_5 = None
        getitem_76 = rel_h_5[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_5 = None
        getitem_77 = rel_w_5[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_5 = None
        decomposed_rel_pos_10 = getitem_76 + getitem_77
        getitem_76 = getitem_77 = None
        decomposed_rel_pos_11 = decomposed_rel_pos_10.reshape(1, 12, 4096, 4096)
        decomposed_rel_pos_10 = None
        query_11 = query_10.view(1, 12, 4096, -1)
        query_10 = None
        key_11 = key_10.view(1, 12, 4096, -1)
        key_10 = None
        value_11 = value_10.view(1, 12, 4096, -1)
        value_10 = None
        attn_output_15 = torch._C._nn.scaled_dot_product_attention(
            query_11, key_11, value_11, attn_mask=decomposed_rel_pos_11
        )
        query_11 = key_11 = value_11 = decomposed_rel_pos_11 = None
        view_23 = attn_output_15.view(1, 12, 64, 64, -1)
        attn_output_15 = None
        permute_45 = view_23.permute(0, 2, 3, 1, 4)
        view_23 = None
        attn_output_16 = permute_45.reshape(1, 64, 64, -1)
        permute_45 = None
        attn_output_17 = torch._C._nn.linear(
            attn_output_16,
            l_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_16 = l_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_52 = hidden_states_50 + attn_output_17
        hidden_states_50 = attn_output_17 = None
        item_12 = (
            l_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_5 = torch.nn.functional.layer_norm(
            hidden_states_52,
            (768,),
            l_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_bias_,
            item_12,
        )
        l_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_bias_ = (item_12) = (
            None
        )
        hidden_states_53 = torch._C._nn.linear(
            layernorm_output_5,
            l_self_modules_vision_encoder_modules_layers_modules_5_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_5_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_5 = l_self_modules_vision_encoder_modules_layers_modules_5_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_5_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_54 = torch._C._nn.gelu(hidden_states_53)
        hidden_states_53 = None
        hidden_states_55 = torch._C._nn.linear(
            hidden_states_54,
            l_self_modules_vision_encoder_modules_layers_modules_5_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_5_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_54 = l_self_modules_vision_encoder_modules_layers_modules_5_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_5_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_56 = hidden_states_52 + hidden_states_55
        hidden_states_52 = hidden_states_55 = None
        item_13 = (
            l_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm1_eps = (
            None
        )
        hidden_states_57 = torch.nn.functional.layer_norm(
            hidden_states_56,
            (768,),
            l_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_bias_,
            item_13,
        )
        l_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_bias_ = (item_13) = (
            None
        )
        hidden_states_58 = torch._C._nn.pad(
            hidden_states_57, (0, 0, 0, 6, 0, 6), "constant", None
        )
        hidden_states_57 = None
        hidden_states_59 = hidden_states_58.reshape(1, 5, 14, 5, 14, 768)
        hidden_states_58 = None
        permute_46 = hidden_states_59.permute(0, 1, 3, 2, 4, 5)
        hidden_states_59 = None
        contiguous_12 = permute_46.contiguous()
        permute_46 = None
        windows_4 = contiguous_12.reshape(-1, 14, 14, 768)
        contiguous_12 = None
        linear_24 = torch._C._nn.linear(
            windows_4,
            l_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_modules_qkv_parameters_bias_,
        )
        windows_4 = l_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_72 = linear_24.reshape(25, 196, 3, 12, -1)
        linear_24 = None
        qkv_6 = reshape_72.permute(2, 0, 3, 1, 4)
        reshape_72 = None
        reshape_73 = qkv_6.reshape(3, 300, 196, -1)
        qkv_6 = None
        unbind_6 = reshape_73.unbind(0)
        reshape_73 = None
        query_12 = unbind_6[0]
        key_12 = unbind_6[1]
        value_12 = unbind_6[2]
        unbind_6 = None
        reshape_74 = l_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_parameters_rel_pos_h_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_48 = reshape_74.permute(0, 2, 1)
        reshape_74 = None
        rel_pos_resized_24 = torch.nn.functional.interpolate(
            permute_48, size=27, mode="linear"
        )
        permute_48 = None
        reshape_75 = rel_pos_resized_24.reshape(-1, 27)
        rel_pos_resized_24 = None
        rel_pos_resized_25 = reshape_75.permute(1, 0)
        reshape_75 = None
        arange_24 = torch.arange(14)
        getitem_81 = arange_24[(slice(None, None, None), None)]
        arange_24 = None
        q_coords_12 = getitem_81 * 1.0
        getitem_81 = None
        arange_25 = torch.arange(14)
        getitem_82 = arange_25[(None, slice(None, None, None))]
        arange_25 = None
        k_coords_12 = getitem_82 * 1.0
        getitem_82 = None
        sub_15 = q_coords_12 - k_coords_12
        q_coords_12 = k_coords_12 = None
        relative_coords_12 = sub_15 + 13.0
        sub_15 = None
        long_12 = relative_coords_12.long()
        relative_coords_12 = None
        relative_position_height_6 = rel_pos_resized_25[long_12]
        rel_pos_resized_25 = long_12 = None
        reshape_76 = l_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_parameters_rel_pos_w_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_50 = reshape_76.permute(0, 2, 1)
        reshape_76 = None
        rel_pos_resized_26 = torch.nn.functional.interpolate(
            permute_50, size=27, mode="linear"
        )
        permute_50 = None
        reshape_77 = rel_pos_resized_26.reshape(-1, 27)
        rel_pos_resized_26 = None
        rel_pos_resized_27 = reshape_77.permute(1, 0)
        reshape_77 = None
        arange_26 = torch.arange(14)
        getitem_84 = arange_26[(slice(None, None, None), None)]
        arange_26 = None
        q_coords_13 = getitem_84 * 1.0
        getitem_84 = None
        arange_27 = torch.arange(14)
        getitem_85 = arange_27[(None, slice(None, None, None))]
        arange_27 = None
        k_coords_13 = getitem_85 * 1.0
        getitem_85 = None
        sub_16 = q_coords_13 - k_coords_13
        q_coords_13 = k_coords_13 = None
        relative_coords_13 = sub_16 + 13.0
        sub_16 = None
        long_13 = relative_coords_13.long()
        relative_coords_13 = None
        relative_position_width_6 = rel_pos_resized_27[long_13]
        rel_pos_resized_27 = long_13 = None
        reshaped_query_6 = query_12.reshape(300, 14, 14, 64)
        rel_h_6 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_6, relative_position_height_6
        )
        relative_position_height_6 = None
        rel_w_6 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_6, relative_position_width_6
        )
        reshaped_query_6 = relative_position_width_6 = None
        getitem_87 = rel_h_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_6 = None
        getitem_88 = rel_w_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_6 = None
        decomposed_rel_pos_12 = getitem_87 + getitem_88
        getitem_87 = getitem_88 = None
        decomposed_rel_pos_13 = decomposed_rel_pos_12.reshape(25, 12, 196, 196)
        decomposed_rel_pos_12 = None
        query_13 = query_12.view(25, 12, 196, -1)
        query_12 = None
        key_13 = key_12.view(25, 12, 196, -1)
        key_12 = None
        value_13 = value_12.view(25, 12, 196, -1)
        value_12 = None
        attn_output_18 = torch._C._nn.scaled_dot_product_attention(
            query_13, key_13, value_13, attn_mask=decomposed_rel_pos_13
        )
        query_13 = key_13 = value_13 = decomposed_rel_pos_13 = None
        view_27 = attn_output_18.view(25, 12, 14, 14, -1)
        attn_output_18 = None
        permute_52 = view_27.permute(0, 2, 3, 1, 4)
        view_27 = None
        attn_output_19 = permute_52.reshape(25, 14, 14, -1)
        permute_52 = None
        attn_output_20 = torch._C._nn.linear(
            attn_output_19,
            l_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_19 = l_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_60 = attn_output_20.reshape(1, 5, 5, 14, 14, -1)
        attn_output_20 = None
        permute_53 = hidden_states_60.permute(0, 1, 3, 2, 4, 5)
        hidden_states_60 = None
        contiguous_13 = permute_53.contiguous()
        permute_53 = None
        hidden_states_61 = contiguous_13.reshape(1, 70, 70, -1)
        contiguous_13 = None
        getitem_89 = hidden_states_61[
            (
                slice(None, None, None),
                slice(None, 64, None),
                slice(None, 64, None),
                slice(None, None, None),
            )
        ]
        hidden_states_61 = None
        hidden_states_62 = getitem_89.contiguous()
        getitem_89 = None
        hidden_states_63 = hidden_states_56 + hidden_states_62
        hidden_states_56 = hidden_states_62 = None
        item_14 = (
            l_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_6 = torch.nn.functional.layer_norm(
            hidden_states_63,
            (768,),
            l_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_bias_,
            item_14,
        )
        l_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_bias_ = (item_14) = (
            None
        )
        hidden_states_64 = torch._C._nn.linear(
            layernorm_output_6,
            l_self_modules_vision_encoder_modules_layers_modules_6_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_6_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_6 = l_self_modules_vision_encoder_modules_layers_modules_6_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_6_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_65 = torch._C._nn.gelu(hidden_states_64)
        hidden_states_64 = None
        hidden_states_66 = torch._C._nn.linear(
            hidden_states_65,
            l_self_modules_vision_encoder_modules_layers_modules_6_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_6_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_65 = l_self_modules_vision_encoder_modules_layers_modules_6_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_6_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_67 = hidden_states_63 + hidden_states_66
        hidden_states_63 = hidden_states_66 = None
        item_15 = (
            l_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm1_eps = (
            None
        )
        hidden_states_68 = torch.nn.functional.layer_norm(
            hidden_states_67,
            (768,),
            l_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_bias_,
            item_15,
        )
        l_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_bias_ = (item_15) = (
            None
        )
        hidden_states_69 = torch._C._nn.pad(
            hidden_states_68, (0, 0, 0, 6, 0, 6), "constant", None
        )
        hidden_states_68 = None
        hidden_states_70 = hidden_states_69.reshape(1, 5, 14, 5, 14, 768)
        hidden_states_69 = None
        permute_54 = hidden_states_70.permute(0, 1, 3, 2, 4, 5)
        hidden_states_70 = None
        contiguous_15 = permute_54.contiguous()
        permute_54 = None
        windows_5 = contiguous_15.reshape(-1, 14, 14, 768)
        contiguous_15 = None
        linear_28 = torch._C._nn.linear(
            windows_5,
            l_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_modules_qkv_parameters_bias_,
        )
        windows_5 = l_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_85 = linear_28.reshape(25, 196, 3, 12, -1)
        linear_28 = None
        qkv_7 = reshape_85.permute(2, 0, 3, 1, 4)
        reshape_85 = None
        reshape_86 = qkv_7.reshape(3, 300, 196, -1)
        qkv_7 = None
        unbind_7 = reshape_86.unbind(0)
        reshape_86 = None
        query_14 = unbind_7[0]
        key_14 = unbind_7[1]
        value_14 = unbind_7[2]
        unbind_7 = None
        reshape_87 = l_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_parameters_rel_pos_h_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_56 = reshape_87.permute(0, 2, 1)
        reshape_87 = None
        rel_pos_resized_28 = torch.nn.functional.interpolate(
            permute_56, size=27, mode="linear"
        )
        permute_56 = None
        reshape_88 = rel_pos_resized_28.reshape(-1, 27)
        rel_pos_resized_28 = None
        rel_pos_resized_29 = reshape_88.permute(1, 0)
        reshape_88 = None
        arange_28 = torch.arange(14)
        getitem_93 = arange_28[(slice(None, None, None), None)]
        arange_28 = None
        q_coords_14 = getitem_93 * 1.0
        getitem_93 = None
        arange_29 = torch.arange(14)
        getitem_94 = arange_29[(None, slice(None, None, None))]
        arange_29 = None
        k_coords_14 = getitem_94 * 1.0
        getitem_94 = None
        sub_17 = q_coords_14 - k_coords_14
        q_coords_14 = k_coords_14 = None
        relative_coords_14 = sub_17 + 13.0
        sub_17 = None
        long_14 = relative_coords_14.long()
        relative_coords_14 = None
        relative_position_height_7 = rel_pos_resized_29[long_14]
        rel_pos_resized_29 = long_14 = None
        reshape_89 = l_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_parameters_rel_pos_w_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_58 = reshape_89.permute(0, 2, 1)
        reshape_89 = None
        rel_pos_resized_30 = torch.nn.functional.interpolate(
            permute_58, size=27, mode="linear"
        )
        permute_58 = None
        reshape_90 = rel_pos_resized_30.reshape(-1, 27)
        rel_pos_resized_30 = None
        rel_pos_resized_31 = reshape_90.permute(1, 0)
        reshape_90 = None
        arange_30 = torch.arange(14)
        getitem_96 = arange_30[(slice(None, None, None), None)]
        arange_30 = None
        q_coords_15 = getitem_96 * 1.0
        getitem_96 = None
        arange_31 = torch.arange(14)
        getitem_97 = arange_31[(None, slice(None, None, None))]
        arange_31 = None
        k_coords_15 = getitem_97 * 1.0
        getitem_97 = None
        sub_18 = q_coords_15 - k_coords_15
        q_coords_15 = k_coords_15 = None
        relative_coords_15 = sub_18 + 13.0
        sub_18 = None
        long_15 = relative_coords_15.long()
        relative_coords_15 = None
        relative_position_width_7 = rel_pos_resized_31[long_15]
        rel_pos_resized_31 = long_15 = None
        reshaped_query_7 = query_14.reshape(300, 14, 14, 64)
        rel_h_7 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_7, relative_position_height_7
        )
        relative_position_height_7 = None
        rel_w_7 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_7, relative_position_width_7
        )
        reshaped_query_7 = relative_position_width_7 = None
        getitem_99 = rel_h_7[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_7 = None
        getitem_100 = rel_w_7[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_7 = None
        decomposed_rel_pos_14 = getitem_99 + getitem_100
        getitem_99 = getitem_100 = None
        decomposed_rel_pos_15 = decomposed_rel_pos_14.reshape(25, 12, 196, 196)
        decomposed_rel_pos_14 = None
        query_15 = query_14.view(25, 12, 196, -1)
        query_14 = None
        key_15 = key_14.view(25, 12, 196, -1)
        key_14 = None
        value_15 = value_14.view(25, 12, 196, -1)
        value_14 = None
        attn_output_21 = torch._C._nn.scaled_dot_product_attention(
            query_15, key_15, value_15, attn_mask=decomposed_rel_pos_15
        )
        query_15 = key_15 = value_15 = decomposed_rel_pos_15 = None
        view_31 = attn_output_21.view(25, 12, 14, 14, -1)
        attn_output_21 = None
        permute_60 = view_31.permute(0, 2, 3, 1, 4)
        view_31 = None
        attn_output_22 = permute_60.reshape(25, 14, 14, -1)
        permute_60 = None
        attn_output_23 = torch._C._nn.linear(
            attn_output_22,
            l_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_22 = l_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_71 = attn_output_23.reshape(1, 5, 5, 14, 14, -1)
        attn_output_23 = None
        permute_61 = hidden_states_71.permute(0, 1, 3, 2, 4, 5)
        hidden_states_71 = None
        contiguous_16 = permute_61.contiguous()
        permute_61 = None
        hidden_states_72 = contiguous_16.reshape(1, 70, 70, -1)
        contiguous_16 = None
        getitem_101 = hidden_states_72[
            (
                slice(None, None, None),
                slice(None, 64, None),
                slice(None, 64, None),
                slice(None, None, None),
            )
        ]
        hidden_states_72 = None
        hidden_states_73 = getitem_101.contiguous()
        getitem_101 = None
        hidden_states_74 = hidden_states_67 + hidden_states_73
        hidden_states_67 = hidden_states_73 = None
        item_16 = (
            l_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_7 = torch.nn.functional.layer_norm(
            hidden_states_74,
            (768,),
            l_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_bias_,
            item_16,
        )
        l_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_bias_ = (item_16) = (
            None
        )
        hidden_states_75 = torch._C._nn.linear(
            layernorm_output_7,
            l_self_modules_vision_encoder_modules_layers_modules_7_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_7_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_7 = l_self_modules_vision_encoder_modules_layers_modules_7_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_7_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_76 = torch._C._nn.gelu(hidden_states_75)
        hidden_states_75 = None
        hidden_states_77 = torch._C._nn.linear(
            hidden_states_76,
            l_self_modules_vision_encoder_modules_layers_modules_7_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_7_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_76 = l_self_modules_vision_encoder_modules_layers_modules_7_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_7_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_78 = hidden_states_74 + hidden_states_77
        hidden_states_74 = hidden_states_77 = None
        item_17 = (
            l_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm1_eps = (
            None
        )
        hidden_states_79 = torch.nn.functional.layer_norm(
            hidden_states_78,
            (768,),
            l_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_bias_,
            item_17,
        )
        l_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_bias_ = (item_17) = (
            None
        )
        linear_32 = torch._C._nn.linear(
            hidden_states_79,
            l_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_modules_qkv_parameters_bias_,
        )
        hidden_states_79 = l_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_96 = linear_32.reshape(1, 4096, 3, 12, -1)
        linear_32 = None
        qkv_8 = reshape_96.permute(2, 0, 3, 1, 4)
        reshape_96 = None
        reshape_97 = qkv_8.reshape(3, 12, 4096, -1)
        qkv_8 = None
        unbind_8 = reshape_97.unbind(0)
        reshape_97 = None
        query_16 = unbind_8[0]
        key_16 = unbind_8[1]
        value_16 = unbind_8[2]
        unbind_8 = None
        reshape_98 = l_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_parameters_rel_pos_h_.reshape(
            1, 127, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_63 = reshape_98.permute(0, 2, 1)
        reshape_98 = None
        rel_pos_resized_32 = torch.nn.functional.interpolate(
            permute_63, size=127, mode="linear"
        )
        permute_63 = None
        reshape_99 = rel_pos_resized_32.reshape(-1, 127)
        rel_pos_resized_32 = None
        rel_pos_resized_33 = reshape_99.permute(1, 0)
        reshape_99 = None
        arange_32 = torch.arange(64)
        getitem_105 = arange_32[(slice(None, None, None), None)]
        arange_32 = None
        q_coords_16 = getitem_105 * 1.0
        getitem_105 = None
        arange_33 = torch.arange(64)
        getitem_106 = arange_33[(None, slice(None, None, None))]
        arange_33 = None
        k_coords_16 = getitem_106 * 1.0
        getitem_106 = None
        sub_19 = q_coords_16 - k_coords_16
        q_coords_16 = k_coords_16 = None
        relative_coords_16 = sub_19 + 63.0
        sub_19 = None
        long_16 = relative_coords_16.long()
        relative_coords_16 = None
        relative_position_height_8 = rel_pos_resized_33[long_16]
        rel_pos_resized_33 = long_16 = None
        reshape_100 = l_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_parameters_rel_pos_w_.reshape(
            1, 127, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_65 = reshape_100.permute(0, 2, 1)
        reshape_100 = None
        rel_pos_resized_34 = torch.nn.functional.interpolate(
            permute_65, size=127, mode="linear"
        )
        permute_65 = None
        reshape_101 = rel_pos_resized_34.reshape(-1, 127)
        rel_pos_resized_34 = None
        rel_pos_resized_35 = reshape_101.permute(1, 0)
        reshape_101 = None
        arange_34 = torch.arange(64)
        getitem_108 = arange_34[(slice(None, None, None), None)]
        arange_34 = None
        q_coords_17 = getitem_108 * 1.0
        getitem_108 = None
        arange_35 = torch.arange(64)
        getitem_109 = arange_35[(None, slice(None, None, None))]
        arange_35 = None
        k_coords_17 = getitem_109 * 1.0
        getitem_109 = None
        sub_20 = q_coords_17 - k_coords_17
        q_coords_17 = k_coords_17 = None
        relative_coords_17 = sub_20 + 63.0
        sub_20 = None
        long_17 = relative_coords_17.long()
        relative_coords_17 = None
        relative_position_width_8 = rel_pos_resized_35[long_17]
        rel_pos_resized_35 = long_17 = None
        reshaped_query_8 = query_16.reshape(12, 64, 64, 64)
        rel_h_8 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_8, relative_position_height_8
        )
        relative_position_height_8 = None
        rel_w_8 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_8, relative_position_width_8
        )
        reshaped_query_8 = relative_position_width_8 = None
        getitem_111 = rel_h_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_8 = None
        getitem_112 = rel_w_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_8 = None
        decomposed_rel_pos_16 = getitem_111 + getitem_112
        getitem_111 = getitem_112 = None
        decomposed_rel_pos_17 = decomposed_rel_pos_16.reshape(1, 12, 4096, 4096)
        decomposed_rel_pos_16 = None
        query_17 = query_16.view(1, 12, 4096, -1)
        query_16 = None
        key_17 = key_16.view(1, 12, 4096, -1)
        key_16 = None
        value_17 = value_16.view(1, 12, 4096, -1)
        value_16 = None
        attn_output_24 = torch._C._nn.scaled_dot_product_attention(
            query_17, key_17, value_17, attn_mask=decomposed_rel_pos_17
        )
        query_17 = key_17 = value_17 = decomposed_rel_pos_17 = None
        view_35 = attn_output_24.view(1, 12, 64, 64, -1)
        attn_output_24 = None
        permute_67 = view_35.permute(0, 2, 3, 1, 4)
        view_35 = None
        attn_output_25 = permute_67.reshape(1, 64, 64, -1)
        permute_67 = None
        attn_output_26 = torch._C._nn.linear(
            attn_output_25,
            l_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_25 = l_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_80 = hidden_states_78 + attn_output_26
        hidden_states_78 = attn_output_26 = None
        item_18 = (
            l_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_8 = torch.nn.functional.layer_norm(
            hidden_states_80,
            (768,),
            l_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_bias_,
            item_18,
        )
        l_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_bias_ = (item_18) = (
            None
        )
        hidden_states_81 = torch._C._nn.linear(
            layernorm_output_8,
            l_self_modules_vision_encoder_modules_layers_modules_8_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_8_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_8 = l_self_modules_vision_encoder_modules_layers_modules_8_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_8_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_82 = torch._C._nn.gelu(hidden_states_81)
        hidden_states_81 = None
        hidden_states_83 = torch._C._nn.linear(
            hidden_states_82,
            l_self_modules_vision_encoder_modules_layers_modules_8_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_8_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_82 = l_self_modules_vision_encoder_modules_layers_modules_8_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_8_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_84 = hidden_states_80 + hidden_states_83
        hidden_states_80 = hidden_states_83 = None
        item_19 = (
            l_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm1_eps = (
            None
        )
        hidden_states_85 = torch.nn.functional.layer_norm(
            hidden_states_84,
            (768,),
            l_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_bias_,
            item_19,
        )
        l_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_bias_ = (item_19) = (
            None
        )
        hidden_states_86 = torch._C._nn.pad(
            hidden_states_85, (0, 0, 0, 6, 0, 6), "constant", None
        )
        hidden_states_85 = None
        hidden_states_87 = hidden_states_86.reshape(1, 5, 14, 5, 14, 768)
        hidden_states_86 = None
        permute_68 = hidden_states_87.permute(0, 1, 3, 2, 4, 5)
        hidden_states_87 = None
        contiguous_18 = permute_68.contiguous()
        permute_68 = None
        windows_6 = contiguous_18.reshape(-1, 14, 14, 768)
        contiguous_18 = None
        linear_36 = torch._C._nn.linear(
            windows_6,
            l_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_modules_qkv_parameters_bias_,
        )
        windows_6 = l_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_107 = linear_36.reshape(25, 196, 3, 12, -1)
        linear_36 = None
        qkv_9 = reshape_107.permute(2, 0, 3, 1, 4)
        reshape_107 = None
        reshape_108 = qkv_9.reshape(3, 300, 196, -1)
        qkv_9 = None
        unbind_9 = reshape_108.unbind(0)
        reshape_108 = None
        query_18 = unbind_9[0]
        key_18 = unbind_9[1]
        value_18 = unbind_9[2]
        unbind_9 = None
        reshape_109 = l_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_parameters_rel_pos_h_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_70 = reshape_109.permute(0, 2, 1)
        reshape_109 = None
        rel_pos_resized_36 = torch.nn.functional.interpolate(
            permute_70, size=27, mode="linear"
        )
        permute_70 = None
        reshape_110 = rel_pos_resized_36.reshape(-1, 27)
        rel_pos_resized_36 = None
        rel_pos_resized_37 = reshape_110.permute(1, 0)
        reshape_110 = None
        arange_36 = torch.arange(14)
        getitem_116 = arange_36[(slice(None, None, None), None)]
        arange_36 = None
        q_coords_18 = getitem_116 * 1.0
        getitem_116 = None
        arange_37 = torch.arange(14)
        getitem_117 = arange_37[(None, slice(None, None, None))]
        arange_37 = None
        k_coords_18 = getitem_117 * 1.0
        getitem_117 = None
        sub_21 = q_coords_18 - k_coords_18
        q_coords_18 = k_coords_18 = None
        relative_coords_18 = sub_21 + 13.0
        sub_21 = None
        long_18 = relative_coords_18.long()
        relative_coords_18 = None
        relative_position_height_9 = rel_pos_resized_37[long_18]
        rel_pos_resized_37 = long_18 = None
        reshape_111 = l_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_parameters_rel_pos_w_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_72 = reshape_111.permute(0, 2, 1)
        reshape_111 = None
        rel_pos_resized_38 = torch.nn.functional.interpolate(
            permute_72, size=27, mode="linear"
        )
        permute_72 = None
        reshape_112 = rel_pos_resized_38.reshape(-1, 27)
        rel_pos_resized_38 = None
        rel_pos_resized_39 = reshape_112.permute(1, 0)
        reshape_112 = None
        arange_38 = torch.arange(14)
        getitem_119 = arange_38[(slice(None, None, None), None)]
        arange_38 = None
        q_coords_19 = getitem_119 * 1.0
        getitem_119 = None
        arange_39 = torch.arange(14)
        getitem_120 = arange_39[(None, slice(None, None, None))]
        arange_39 = None
        k_coords_19 = getitem_120 * 1.0
        getitem_120 = None
        sub_22 = q_coords_19 - k_coords_19
        q_coords_19 = k_coords_19 = None
        relative_coords_19 = sub_22 + 13.0
        sub_22 = None
        long_19 = relative_coords_19.long()
        relative_coords_19 = None
        relative_position_width_9 = rel_pos_resized_39[long_19]
        rel_pos_resized_39 = long_19 = None
        reshaped_query_9 = query_18.reshape(300, 14, 14, 64)
        rel_h_9 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_9, relative_position_height_9
        )
        relative_position_height_9 = None
        rel_w_9 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_9, relative_position_width_9
        )
        reshaped_query_9 = relative_position_width_9 = None
        getitem_122 = rel_h_9[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_9 = None
        getitem_123 = rel_w_9[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_9 = None
        decomposed_rel_pos_18 = getitem_122 + getitem_123
        getitem_122 = getitem_123 = None
        decomposed_rel_pos_19 = decomposed_rel_pos_18.reshape(25, 12, 196, 196)
        decomposed_rel_pos_18 = None
        query_19 = query_18.view(25, 12, 196, -1)
        query_18 = None
        key_19 = key_18.view(25, 12, 196, -1)
        key_18 = None
        value_19 = value_18.view(25, 12, 196, -1)
        value_18 = None
        attn_output_27 = torch._C._nn.scaled_dot_product_attention(
            query_19, key_19, value_19, attn_mask=decomposed_rel_pos_19
        )
        query_19 = key_19 = value_19 = decomposed_rel_pos_19 = None
        view_39 = attn_output_27.view(25, 12, 14, 14, -1)
        attn_output_27 = None
        permute_74 = view_39.permute(0, 2, 3, 1, 4)
        view_39 = None
        attn_output_28 = permute_74.reshape(25, 14, 14, -1)
        permute_74 = None
        attn_output_29 = torch._C._nn.linear(
            attn_output_28,
            l_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_28 = l_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_88 = attn_output_29.reshape(1, 5, 5, 14, 14, -1)
        attn_output_29 = None
        permute_75 = hidden_states_88.permute(0, 1, 3, 2, 4, 5)
        hidden_states_88 = None
        contiguous_19 = permute_75.contiguous()
        permute_75 = None
        hidden_states_89 = contiguous_19.reshape(1, 70, 70, -1)
        contiguous_19 = None
        getitem_124 = hidden_states_89[
            (
                slice(None, None, None),
                slice(None, 64, None),
                slice(None, 64, None),
                slice(None, None, None),
            )
        ]
        hidden_states_89 = None
        hidden_states_90 = getitem_124.contiguous()
        getitem_124 = None
        hidden_states_91 = hidden_states_84 + hidden_states_90
        hidden_states_84 = hidden_states_90 = None
        item_20 = (
            l_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_9 = torch.nn.functional.layer_norm(
            hidden_states_91,
            (768,),
            l_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_bias_,
            item_20,
        )
        l_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_bias_ = (item_20) = (
            None
        )
        hidden_states_92 = torch._C._nn.linear(
            layernorm_output_9,
            l_self_modules_vision_encoder_modules_layers_modules_9_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_9_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_9 = l_self_modules_vision_encoder_modules_layers_modules_9_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_9_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_93 = torch._C._nn.gelu(hidden_states_92)
        hidden_states_92 = None
        hidden_states_94 = torch._C._nn.linear(
            hidden_states_93,
            l_self_modules_vision_encoder_modules_layers_modules_9_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_9_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_93 = l_self_modules_vision_encoder_modules_layers_modules_9_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_9_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_95 = hidden_states_91 + hidden_states_94
        hidden_states_91 = hidden_states_94 = None
        item_21 = (
            l_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm1_eps = (
            None
        )
        hidden_states_96 = torch.nn.functional.layer_norm(
            hidden_states_95,
            (768,),
            l_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_bias_,
            item_21,
        )
        l_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_bias_ = (item_21) = (
            None
        )
        hidden_states_97 = torch._C._nn.pad(
            hidden_states_96, (0, 0, 0, 6, 0, 6), "constant", None
        )
        hidden_states_96 = None
        hidden_states_98 = hidden_states_97.reshape(1, 5, 14, 5, 14, 768)
        hidden_states_97 = None
        permute_76 = hidden_states_98.permute(0, 1, 3, 2, 4, 5)
        hidden_states_98 = None
        contiguous_21 = permute_76.contiguous()
        permute_76 = None
        windows_7 = contiguous_21.reshape(-1, 14, 14, 768)
        contiguous_21 = None
        linear_40 = torch._C._nn.linear(
            windows_7,
            l_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_modules_qkv_parameters_bias_,
        )
        windows_7 = l_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_120 = linear_40.reshape(25, 196, 3, 12, -1)
        linear_40 = None
        qkv_10 = reshape_120.permute(2, 0, 3, 1, 4)
        reshape_120 = None
        reshape_121 = qkv_10.reshape(3, 300, 196, -1)
        qkv_10 = None
        unbind_10 = reshape_121.unbind(0)
        reshape_121 = None
        query_20 = unbind_10[0]
        key_20 = unbind_10[1]
        value_20 = unbind_10[2]
        unbind_10 = None
        reshape_122 = l_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_parameters_rel_pos_h_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_78 = reshape_122.permute(0, 2, 1)
        reshape_122 = None
        rel_pos_resized_40 = torch.nn.functional.interpolate(
            permute_78, size=27, mode="linear"
        )
        permute_78 = None
        reshape_123 = rel_pos_resized_40.reshape(-1, 27)
        rel_pos_resized_40 = None
        rel_pos_resized_41 = reshape_123.permute(1, 0)
        reshape_123 = None
        arange_40 = torch.arange(14)
        getitem_128 = arange_40[(slice(None, None, None), None)]
        arange_40 = None
        q_coords_20 = getitem_128 * 1.0
        getitem_128 = None
        arange_41 = torch.arange(14)
        getitem_129 = arange_41[(None, slice(None, None, None))]
        arange_41 = None
        k_coords_20 = getitem_129 * 1.0
        getitem_129 = None
        sub_23 = q_coords_20 - k_coords_20
        q_coords_20 = k_coords_20 = None
        relative_coords_20 = sub_23 + 13.0
        sub_23 = None
        long_20 = relative_coords_20.long()
        relative_coords_20 = None
        relative_position_height_10 = rel_pos_resized_41[long_20]
        rel_pos_resized_41 = long_20 = None
        reshape_124 = l_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_parameters_rel_pos_w_.reshape(
            1, 27, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_80 = reshape_124.permute(0, 2, 1)
        reshape_124 = None
        rel_pos_resized_42 = torch.nn.functional.interpolate(
            permute_80, size=27, mode="linear"
        )
        permute_80 = None
        reshape_125 = rel_pos_resized_42.reshape(-1, 27)
        rel_pos_resized_42 = None
        rel_pos_resized_43 = reshape_125.permute(1, 0)
        reshape_125 = None
        arange_42 = torch.arange(14)
        getitem_131 = arange_42[(slice(None, None, None), None)]
        arange_42 = None
        q_coords_21 = getitem_131 * 1.0
        getitem_131 = None
        arange_43 = torch.arange(14)
        getitem_132 = arange_43[(None, slice(None, None, None))]
        arange_43 = None
        k_coords_21 = getitem_132 * 1.0
        getitem_132 = None
        sub_24 = q_coords_21 - k_coords_21
        q_coords_21 = k_coords_21 = None
        relative_coords_21 = sub_24 + 13.0
        sub_24 = None
        long_21 = relative_coords_21.long()
        relative_coords_21 = None
        relative_position_width_10 = rel_pos_resized_43[long_21]
        rel_pos_resized_43 = long_21 = None
        reshaped_query_10 = query_20.reshape(300, 14, 14, 64)
        rel_h_10 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_10, relative_position_height_10
        )
        relative_position_height_10 = None
        rel_w_10 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_10, relative_position_width_10
        )
        reshaped_query_10 = relative_position_width_10 = None
        getitem_134 = rel_h_10[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_10 = None
        getitem_135 = rel_w_10[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_10 = None
        decomposed_rel_pos_20 = getitem_134 + getitem_135
        getitem_134 = getitem_135 = None
        decomposed_rel_pos_21 = decomposed_rel_pos_20.reshape(25, 12, 196, 196)
        decomposed_rel_pos_20 = None
        query_21 = query_20.view(25, 12, 196, -1)
        query_20 = None
        key_21 = key_20.view(25, 12, 196, -1)
        key_20 = None
        value_21 = value_20.view(25, 12, 196, -1)
        value_20 = None
        attn_output_30 = torch._C._nn.scaled_dot_product_attention(
            query_21, key_21, value_21, attn_mask=decomposed_rel_pos_21
        )
        query_21 = key_21 = value_21 = decomposed_rel_pos_21 = None
        view_43 = attn_output_30.view(25, 12, 14, 14, -1)
        attn_output_30 = None
        permute_82 = view_43.permute(0, 2, 3, 1, 4)
        view_43 = None
        attn_output_31 = permute_82.reshape(25, 14, 14, -1)
        permute_82 = None
        attn_output_32 = torch._C._nn.linear(
            attn_output_31,
            l_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_31 = l_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_99 = attn_output_32.reshape(1, 5, 5, 14, 14, -1)
        attn_output_32 = None
        permute_83 = hidden_states_99.permute(0, 1, 3, 2, 4, 5)
        hidden_states_99 = None
        contiguous_22 = permute_83.contiguous()
        permute_83 = None
        hidden_states_100 = contiguous_22.reshape(1, 70, 70, -1)
        contiguous_22 = None
        getitem_136 = hidden_states_100[
            (
                slice(None, None, None),
                slice(None, 64, None),
                slice(None, 64, None),
                slice(None, None, None),
            )
        ]
        hidden_states_100 = None
        hidden_states_101 = getitem_136.contiguous()
        getitem_136 = None
        hidden_states_102 = hidden_states_95 + hidden_states_101
        hidden_states_95 = hidden_states_101 = None
        item_22 = (
            l_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_10 = torch.nn.functional.layer_norm(
            hidden_states_102,
            (768,),
            l_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_bias_,
            item_22,
        )
        l_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_bias_ = (item_22) = (
            None
        )
        hidden_states_103 = torch._C._nn.linear(
            layernorm_output_10,
            l_self_modules_vision_encoder_modules_layers_modules_10_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_10_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_10 = l_self_modules_vision_encoder_modules_layers_modules_10_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_10_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_104 = torch._C._nn.gelu(hidden_states_103)
        hidden_states_103 = None
        hidden_states_105 = torch._C._nn.linear(
            hidden_states_104,
            l_self_modules_vision_encoder_modules_layers_modules_10_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_10_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_104 = l_self_modules_vision_encoder_modules_layers_modules_10_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_10_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_106 = hidden_states_102 + hidden_states_105
        hidden_states_102 = hidden_states_105 = None
        item_23 = (
            l_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm1_eps = (
            None
        )
        hidden_states_107 = torch.nn.functional.layer_norm(
            hidden_states_106,
            (768,),
            l_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_bias_,
            item_23,
        )
        l_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_bias_ = (item_23) = (
            None
        )
        linear_44 = torch._C._nn.linear(
            hidden_states_107,
            l_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_modules_qkv_parameters_bias_,
        )
        hidden_states_107 = l_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_modules_qkv_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_modules_qkv_parameters_bias_ = (None)
        reshape_131 = linear_44.reshape(1, 4096, 3, 12, -1)
        linear_44 = None
        qkv_11 = reshape_131.permute(2, 0, 3, 1, 4)
        reshape_131 = None
        reshape_132 = qkv_11.reshape(3, 12, 4096, -1)
        qkv_11 = None
        unbind_11 = reshape_132.unbind(0)
        reshape_132 = None
        query_22 = unbind_11[0]
        key_22 = unbind_11[1]
        value_22 = unbind_11[2]
        unbind_11 = None
        reshape_133 = l_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_parameters_rel_pos_h_.reshape(
            1, 127, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_parameters_rel_pos_h_ = (
            None
        )
        permute_85 = reshape_133.permute(0, 2, 1)
        reshape_133 = None
        rel_pos_resized_44 = torch.nn.functional.interpolate(
            permute_85, size=127, mode="linear"
        )
        permute_85 = None
        reshape_134 = rel_pos_resized_44.reshape(-1, 127)
        rel_pos_resized_44 = None
        rel_pos_resized_45 = reshape_134.permute(1, 0)
        reshape_134 = None
        arange_44 = torch.arange(64)
        getitem_140 = arange_44[(slice(None, None, None), None)]
        arange_44 = None
        q_coords_22 = getitem_140 * 1.0
        getitem_140 = None
        arange_45 = torch.arange(64)
        getitem_141 = arange_45[(None, slice(None, None, None))]
        arange_45 = None
        k_coords_22 = getitem_141 * 1.0
        getitem_141 = None
        sub_25 = q_coords_22 - k_coords_22
        q_coords_22 = k_coords_22 = None
        relative_coords_22 = sub_25 + 63.0
        sub_25 = None
        long_22 = relative_coords_22.long()
        relative_coords_22 = None
        relative_position_height_11 = rel_pos_resized_45[long_22]
        rel_pos_resized_45 = long_22 = None
        reshape_135 = l_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_parameters_rel_pos_w_.reshape(
            1, 127, -1
        )
        l_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_parameters_rel_pos_w_ = (
            None
        )
        permute_87 = reshape_135.permute(0, 2, 1)
        reshape_135 = None
        rel_pos_resized_46 = torch.nn.functional.interpolate(
            permute_87, size=127, mode="linear"
        )
        permute_87 = None
        reshape_136 = rel_pos_resized_46.reshape(-1, 127)
        rel_pos_resized_46 = None
        rel_pos_resized_47 = reshape_136.permute(1, 0)
        reshape_136 = None
        arange_46 = torch.arange(64)
        getitem_143 = arange_46[(slice(None, None, None), None)]
        arange_46 = None
        q_coords_23 = getitem_143 * 1.0
        getitem_143 = None
        arange_47 = torch.arange(64)
        getitem_144 = arange_47[(None, slice(None, None, None))]
        arange_47 = None
        k_coords_23 = getitem_144 * 1.0
        getitem_144 = None
        sub_26 = q_coords_23 - k_coords_23
        q_coords_23 = k_coords_23 = None
        relative_coords_23 = sub_26 + 63.0
        sub_26 = None
        long_23 = relative_coords_23.long()
        relative_coords_23 = None
        relative_position_width_11 = rel_pos_resized_47[long_23]
        rel_pos_resized_47 = long_23 = None
        reshaped_query_11 = query_22.reshape(12, 64, 64, 64)
        rel_h_11 = torch.functional.einsum(
            "bhwc,hkc->bhwk", reshaped_query_11, relative_position_height_11
        )
        relative_position_height_11 = None
        rel_w_11 = torch.functional.einsum(
            "bhwc,wkc->bhwk", reshaped_query_11, relative_position_width_11
        )
        reshaped_query_11 = relative_position_width_11 = None
        getitem_146 = rel_h_11[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_11 = None
        getitem_147 = rel_w_11[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_11 = None
        decomposed_rel_pos_22 = getitem_146 + getitem_147
        getitem_146 = getitem_147 = None
        decomposed_rel_pos_23 = decomposed_rel_pos_22.reshape(1, 12, 4096, 4096)
        decomposed_rel_pos_22 = None
        query_23 = query_22.view(1, 12, 4096, -1)
        query_22 = None
        key_23 = key_22.view(1, 12, 4096, -1)
        key_22 = None
        value_23 = value_22.view(1, 12, 4096, -1)
        value_22 = None
        attn_output_33 = torch._C._nn.scaled_dot_product_attention(
            query_23, key_23, value_23, attn_mask=decomposed_rel_pos_23
        )
        query_23 = key_23 = value_23 = decomposed_rel_pos_23 = None
        view_47 = attn_output_33.view(1, 12, 64, 64, -1)
        attn_output_33 = None
        permute_89 = view_47.permute(0, 2, 3, 1, 4)
        view_47 = None
        attn_output_34 = permute_89.reshape(1, 64, 64, -1)
        permute_89 = None
        attn_output_35 = torch._C._nn.linear(
            attn_output_34,
            l_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_modules_proj_parameters_bias_,
        )
        attn_output_34 = l_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_modules_proj_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_modules_proj_parameters_bias_ = (None)
        hidden_states_108 = hidden_states_106 + attn_output_35
        hidden_states_106 = attn_output_35 = None
        item_24 = (
            l_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm2_eps = (
            None
        )
        layernorm_output_11 = torch.nn.functional.layer_norm(
            hidden_states_108,
            (768,),
            l_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_bias_,
            item_24,
        )
        l_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_bias_ = (item_24) = (
            None
        )
        hidden_states_109 = torch._C._nn.linear(
            layernorm_output_11,
            l_self_modules_vision_encoder_modules_layers_modules_11_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_11_modules_mlp_modules_lin1_parameters_bias_,
        )
        layernorm_output_11 = l_self_modules_vision_encoder_modules_layers_modules_11_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_11_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_110 = torch._C._nn.gelu(hidden_states_109)
        hidden_states_109 = None
        hidden_states_111 = torch._C._nn.linear(
            hidden_states_110,
            l_self_modules_vision_encoder_modules_layers_modules_11_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_vision_encoder_modules_layers_modules_11_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_110 = l_self_modules_vision_encoder_modules_layers_modules_11_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_vision_encoder_modules_layers_modules_11_modules_mlp_modules_lin2_parameters_bias_ = (None)
        hidden_states_112 = hidden_states_108 + hidden_states_111
        hidden_states_108 = hidden_states_111 = None
        hidden_states_113 = hidden_states_112.permute(0, 3, 1, 2)
        hidden_states_112 = None
        hidden_states_114 = torch.conv2d(
            hidden_states_113,
            l_self_modules_vision_encoder_modules_neck_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_113 = (
            l_self_modules_vision_encoder_modules_neck_modules_conv1_parameters_weight_
        ) = None
        x = hidden_states_114.float()
        hidden_states_114 = None
        u = x.mean(1, keepdim=True)
        sub_27 = x - u
        pow_1 = sub_27.pow(2)
        sub_27 = None
        s = pow_1.mean(1, keepdim=True)
        pow_1 = None
        sub_28 = x - u
        x = u = None
        item_25 = (
            l_self_modules_vision_encoder_modules_neck_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_encoder_modules_neck_modules_layer_norm1_eps = None
        add_61 = s + item_25
        s = item_25 = None
        sqrt = torch.sqrt(add_61)
        add_61 = None
        x_1 = sub_28 / sqrt
        sub_28 = sqrt = None
        x_2 = x_1.to(dtype=torch.float32)
        x_1 = None
        getitem_148 = l_self_modules_vision_encoder_modules_neck_modules_layer_norm1_parameters_weight_[
            (slice(None, None, None), None, None)
        ]
        l_self_modules_vision_encoder_modules_neck_modules_layer_norm1_parameters_weight_ = (
            None
        )
        mul_51 = getitem_148 * x_2
        getitem_148 = x_2 = None
        getitem_149 = l_self_modules_vision_encoder_modules_neck_modules_layer_norm1_parameters_bias_[
            (slice(None, None, None), None, None)
        ]
        l_self_modules_vision_encoder_modules_neck_modules_layer_norm1_parameters_bias_ = (
            None
        )
        x_3 = mul_51 + getitem_149
        mul_51 = getitem_149 = None
        hidden_states_115 = torch.conv2d(
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
        x_4 = hidden_states_115.float()
        hidden_states_115 = None
        u_1 = x_4.mean(1, keepdim=True)
        sub_29 = x_4 - u_1
        pow_2 = sub_29.pow(2)
        sub_29 = None
        s_1 = pow_2.mean(1, keepdim=True)
        pow_2 = None
        sub_30 = x_4 - u_1
        x_4 = u_1 = None
        item_26 = (
            l_self_modules_vision_encoder_modules_neck_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_encoder_modules_neck_modules_layer_norm2_eps = None
        add_63 = s_1 + item_26
        s_1 = item_26 = None
        sqrt_1 = torch.sqrt(add_63)
        add_63 = None
        x_5 = sub_30 / sqrt_1
        sub_30 = sqrt_1 = None
        x_6 = x_5.to(dtype=torch.float32)
        x_5 = None
        getitem_150 = l_self_modules_vision_encoder_modules_neck_modules_layer_norm2_parameters_weight_[
            (slice(None, None, None), None, None)
        ]
        l_self_modules_vision_encoder_modules_neck_modules_layer_norm2_parameters_weight_ = (
            None
        )
        mul_52 = getitem_150 * x_6
        getitem_150 = x_6 = None
        getitem_151 = l_self_modules_vision_encoder_modules_neck_modules_layer_norm2_parameters_bias_[
            (slice(None, None, None), None, None)
        ]
        l_self_modules_vision_encoder_modules_neck_modules_layer_norm2_parameters_bias_ = (
            None
        )
        x_7 = mul_52 + getitem_151
        mul_52 = getitem_151 = None
        reshape_140 = l_self_modules_prompt_encoder_modules_no_mask_embed_parameters_weight_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_prompt_encoder_modules_no_mask_embed_parameters_weight_ = None
        dense_embeddings = reshape_140.expand(1, -1, 64, 64)
        reshape_140 = None
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
        permute_91 = flatten.permute(0, 2, 1)
        flatten = None
        image_embeddings_2 = permute_91.unsqueeze(1)
        permute_91 = None
        flatten_1 = image_positional_embeddings_2.flatten(2)
        image_positional_embeddings_2 = None
        permute_92 = flatten_1.permute(0, 2, 1)
        flatten_1 = None
        image_positional_embeddings_3 = permute_92.unsqueeze(1)
        permute_92 = None
        query_24 = torch._C._nn.linear(
            point_embeddings,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        key_24 = torch._C._nn.linear(
            point_embeddings,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_24 = torch._C._nn.linear(
            point_embeddings,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        hidden_states_116 = query_24.reshape(1, 5, 8, 32)
        query_24 = None
        query_25 = hidden_states_116.transpose(1, 2)
        hidden_states_116 = None
        hidden_states_117 = key_24.reshape(1, 5, 8, 32)
        key_24 = None
        key_25 = hidden_states_117.transpose(1, 2)
        hidden_states_117 = None
        hidden_states_118 = value_24.reshape(1, 5, 8, 32)
        value_24 = None
        value_25 = hidden_states_118.transpose(1, 2)
        hidden_states_118 = None
        query_26 = query_25.contiguous()
        query_25 = None
        key_26 = key_25.contiguous()
        key_25 = None
        value_26 = value_25.contiguous()
        value_25 = None
        item_27 = (
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_scaling.item()
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_scaling = (
            None
        )
        attn_output_36 = torch._C._nn.scaled_dot_product_attention(
            query_26,
            key_26,
            value_26,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_27,
            is_causal=False,
        )
        query_26 = key_26 = value_26 = item_27 = None
        transpose_3 = attn_output_36.transpose(1, 2)
        attn_output_36 = None
        attn_output_37 = transpose_3.contiguous()
        transpose_3 = None
        attn_output_38 = attn_output_37.reshape(1, 1, 5, 256)
        attn_output_37 = None
        attn_output_39 = torch._C._nn.linear(
            attn_output_38,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_38 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        item_28 = (
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm1_eps.item()
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm1_eps = (
            None
        )
        queries = torch.nn.functional.layer_norm(
            attn_output_39,
            (256,),
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm1_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm1_parameters_bias_,
            item_28,
        )
        attn_output_39 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm1_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm1_parameters_bias_ = (item_28) = (
            None
        )
        query_27 = queries + point_embeddings
        key_27 = image_embeddings_2 + image_positional_embeddings_3
        query_28 = torch._C._nn.linear(
            query_27,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_q_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_q_proj_parameters_bias_,
        )
        query_27 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_q_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_q_proj_parameters_bias_ = (None)
        key_28 = torch._C._nn.linear(
            key_27,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_k_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_k_proj_parameters_bias_,
        )
        key_27 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_k_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_k_proj_parameters_bias_ = (None)
        value_27 = torch._C._nn.linear(
            image_embeddings_2,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_v_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_v_proj_parameters_bias_,
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_v_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_v_proj_parameters_bias_ = (None)
        hidden_states_119 = query_28.reshape(1, 5, 8, 16)
        query_28 = None
        query_29 = hidden_states_119.transpose(1, 2)
        hidden_states_119 = None
        hidden_states_120 = key_28.reshape(1, 4096, 8, 16)
        key_28 = None
        key_29 = hidden_states_120.transpose(1, 2)
        hidden_states_120 = None
        hidden_states_121 = value_27.reshape(1, 4096, 8, 16)
        value_27 = None
        value_28 = hidden_states_121.transpose(1, 2)
        hidden_states_121 = None
        query_30 = query_29.contiguous()
        query_29 = None
        key_30 = key_29.contiguous()
        key_29 = None
        value_29 = value_28.contiguous()
        value_28 = None
        item_29 = (
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_scaling.item()
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_scaling = (
            None
        )
        attn_output_40 = torch._C._nn.scaled_dot_product_attention(
            query_30,
            key_30,
            value_29,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_29,
            is_causal=False,
        )
        query_30 = key_30 = value_29 = item_29 = None
        transpose_7 = attn_output_40.transpose(1, 2)
        attn_output_40 = None
        attn_output_41 = transpose_7.contiguous()
        transpose_7 = None
        attn_output_42 = attn_output_41.reshape(1, 1, 5, 128)
        attn_output_41 = None
        attn_output_43 = torch._C._nn.linear(
            attn_output_42,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_out_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_out_proj_parameters_bias_,
        )
        attn_output_42 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_out_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_out_proj_parameters_bias_ = (None)
        queries_1 = queries + attn_output_43
        queries = attn_output_43 = None
        item_30 = (
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
            item_30,
        )
        queries_1 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm2_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm2_parameters_bias_ = (item_30) = (
            None
        )
        hidden_states_122 = torch._C._nn.linear(
            queries_2,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_mlp_modules_lin1_parameters_bias_,
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_123 = torch.nn.functional.relu(hidden_states_122, inplace=False)
        hidden_states_122 = None
        hidden_states_124 = torch._C._nn.linear(
            hidden_states_123,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_123 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_mlp_modules_lin2_parameters_bias_ = (None)
        queries_3 = queries_2 + hidden_states_124
        queries_2 = hidden_states_124 = None
        item_31 = (
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
            item_31,
        )
        queries_3 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm3_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm3_parameters_bias_ = (item_31) = (
            None
        )
        query_31 = queries_4 + point_embeddings
        key_31 = image_embeddings_2 + image_positional_embeddings_3
        query_32 = torch._C._nn.linear(
            key_31,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_q_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_q_proj_parameters_bias_,
        )
        key_31 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_q_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_q_proj_parameters_bias_ = (None)
        key_32 = torch._C._nn.linear(
            query_31,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_k_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_k_proj_parameters_bias_,
        )
        query_31 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_k_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_k_proj_parameters_bias_ = (None)
        value_30 = torch._C._nn.linear(
            queries_4,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_v_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_v_proj_parameters_bias_,
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_v_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_v_proj_parameters_bias_ = (None)
        hidden_states_125 = query_32.reshape(1, 4096, 8, 16)
        query_32 = None
        query_33 = hidden_states_125.transpose(1, 2)
        hidden_states_125 = None
        hidden_states_126 = key_32.reshape(1, 5, 8, 16)
        key_32 = None
        key_33 = hidden_states_126.transpose(1, 2)
        hidden_states_126 = None
        hidden_states_127 = value_30.reshape(1, 5, 8, 16)
        value_30 = None
        value_31 = hidden_states_127.transpose(1, 2)
        hidden_states_127 = None
        query_34 = query_33.contiguous()
        query_33 = None
        key_34 = key_33.contiguous()
        key_33 = None
        value_32 = value_31.contiguous()
        value_31 = None
        item_32 = (
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_scaling.item()
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_scaling = (
            None
        )
        attn_output_44 = torch._C._nn.scaled_dot_product_attention(
            query_34,
            key_34,
            value_32,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_32,
            is_causal=False,
        )
        query_34 = key_34 = value_32 = item_32 = None
        transpose_11 = attn_output_44.transpose(1, 2)
        attn_output_44 = None
        attn_output_45 = transpose_11.contiguous()
        transpose_11 = None
        attn_output_46 = attn_output_45.reshape(1, 1, 4096, 128)
        attn_output_45 = None
        attn_output_47 = torch._C._nn.linear(
            attn_output_46,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_out_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_out_proj_parameters_bias_,
        )
        attn_output_46 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_out_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_out_proj_parameters_bias_ = (None)
        keys = image_embeddings_2 + attn_output_47
        image_embeddings_2 = attn_output_47 = None
        item_33 = (
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
            item_33,
        )
        keys = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm4_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm4_parameters_bias_ = (item_33) = (
            None
        )
        query_35 = queries_4 + point_embeddings
        query_36 = torch._C._nn.linear(
            query_35,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        key_35 = torch._C._nn.linear(
            query_35,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        query_35 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_33 = torch._C._nn.linear(
            queries_4,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        hidden_states_128 = query_36.reshape(1, 5, 8, 32)
        query_36 = None
        query_37 = hidden_states_128.transpose(1, 2)
        hidden_states_128 = None
        hidden_states_129 = key_35.reshape(1, 5, 8, 32)
        key_35 = None
        key_36 = hidden_states_129.transpose(1, 2)
        hidden_states_129 = None
        hidden_states_130 = value_33.reshape(1, 5, 8, 32)
        value_33 = None
        value_34 = hidden_states_130.transpose(1, 2)
        hidden_states_130 = None
        query_38 = query_37.contiguous()
        query_37 = None
        key_37 = key_36.contiguous()
        key_36 = None
        value_35 = value_34.contiguous()
        value_34 = None
        item_34 = (
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_scaling.item()
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_scaling = (
            None
        )
        attn_output_48 = torch._C._nn.scaled_dot_product_attention(
            query_38,
            key_37,
            value_35,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_34,
            is_causal=False,
        )
        query_38 = key_37 = value_35 = item_34 = None
        transpose_15 = attn_output_48.transpose(1, 2)
        attn_output_48 = None
        attn_output_49 = transpose_15.contiguous()
        transpose_15 = None
        attn_output_50 = attn_output_49.reshape(1, 1, 5, 256)
        attn_output_49 = None
        attn_output_51 = torch._C._nn.linear(
            attn_output_50,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_50 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        queries_5 = queries_4 + attn_output_51
        queries_4 = attn_output_51 = None
        item_35 = (
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
            item_35,
        )
        queries_5 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm1_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm1_parameters_bias_ = (item_35) = (
            None
        )
        query_39 = queries_6 + point_embeddings
        key_38 = keys_1 + image_positional_embeddings_3
        query_40 = torch._C._nn.linear(
            query_39,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_q_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_q_proj_parameters_bias_,
        )
        query_39 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_q_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_q_proj_parameters_bias_ = (None)
        key_39 = torch._C._nn.linear(
            key_38,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_k_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_k_proj_parameters_bias_,
        )
        key_38 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_k_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_k_proj_parameters_bias_ = (None)
        value_36 = torch._C._nn.linear(
            keys_1,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_v_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_v_proj_parameters_bias_,
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_v_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_v_proj_parameters_bias_ = (None)
        hidden_states_131 = query_40.reshape(1, 5, 8, 16)
        query_40 = None
        query_41 = hidden_states_131.transpose(1, 2)
        hidden_states_131 = None
        hidden_states_132 = key_39.reshape(1, 4096, 8, 16)
        key_39 = None
        key_40 = hidden_states_132.transpose(1, 2)
        hidden_states_132 = None
        hidden_states_133 = value_36.reshape(1, 4096, 8, 16)
        value_36 = None
        value_37 = hidden_states_133.transpose(1, 2)
        hidden_states_133 = None
        query_42 = query_41.contiguous()
        query_41 = None
        key_41 = key_40.contiguous()
        key_40 = None
        value_38 = value_37.contiguous()
        value_37 = None
        item_36 = (
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_scaling.item()
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_scaling = (
            None
        )
        attn_output_52 = torch._C._nn.scaled_dot_product_attention(
            query_42,
            key_41,
            value_38,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_36,
            is_causal=False,
        )
        query_42 = key_41 = value_38 = item_36 = None
        transpose_19 = attn_output_52.transpose(1, 2)
        attn_output_52 = None
        attn_output_53 = transpose_19.contiguous()
        transpose_19 = None
        attn_output_54 = attn_output_53.reshape(1, 1, 5, 128)
        attn_output_53 = None
        attn_output_55 = torch._C._nn.linear(
            attn_output_54,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_out_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_out_proj_parameters_bias_,
        )
        attn_output_54 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_out_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_out_proj_parameters_bias_ = (None)
        queries_7 = queries_6 + attn_output_55
        queries_6 = attn_output_55 = None
        item_37 = (
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
            item_37,
        )
        queries_7 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm2_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm2_parameters_bias_ = (item_37) = (
            None
        )
        hidden_states_134 = torch._C._nn.linear(
            queries_8,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_mlp_modules_lin1_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_mlp_modules_lin1_parameters_bias_,
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_mlp_modules_lin1_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_mlp_modules_lin1_parameters_bias_ = (None)
        hidden_states_135 = torch.nn.functional.relu(hidden_states_134, inplace=False)
        hidden_states_134 = None
        hidden_states_136 = torch._C._nn.linear(
            hidden_states_135,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_mlp_modules_lin2_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_mlp_modules_lin2_parameters_bias_,
        )
        hidden_states_135 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_mlp_modules_lin2_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_mlp_modules_lin2_parameters_bias_ = (None)
        queries_9 = queries_8 + hidden_states_136
        queries_8 = hidden_states_136 = None
        item_38 = (
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
            item_38,
        )
        queries_9 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm3_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm3_parameters_bias_ = (item_38) = (
            None
        )
        query_43 = queries_10 + point_embeddings
        key_42 = keys_1 + image_positional_embeddings_3
        query_44 = torch._C._nn.linear(
            key_42,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_q_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_q_proj_parameters_bias_,
        )
        key_42 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_q_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_q_proj_parameters_bias_ = (None)
        key_43 = torch._C._nn.linear(
            query_43,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_k_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_k_proj_parameters_bias_,
        )
        query_43 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_k_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_k_proj_parameters_bias_ = (None)
        value_39 = torch._C._nn.linear(
            queries_10,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_v_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_v_proj_parameters_bias_,
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_v_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_v_proj_parameters_bias_ = (None)
        hidden_states_137 = query_44.reshape(1, 4096, 8, 16)
        query_44 = None
        query_45 = hidden_states_137.transpose(1, 2)
        hidden_states_137 = None
        hidden_states_138 = key_43.reshape(1, 5, 8, 16)
        key_43 = None
        key_44 = hidden_states_138.transpose(1, 2)
        hidden_states_138 = None
        hidden_states_139 = value_39.reshape(1, 5, 8, 16)
        value_39 = None
        value_40 = hidden_states_139.transpose(1, 2)
        hidden_states_139 = None
        query_46 = query_45.contiguous()
        query_45 = None
        key_45 = key_44.contiguous()
        key_44 = None
        value_41 = value_40.contiguous()
        value_40 = None
        item_39 = (
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_scaling.item()
        )
        l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_scaling = (
            None
        )
        attn_output_56 = torch._C._nn.scaled_dot_product_attention(
            query_46,
            key_45,
            value_41,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_39,
            is_causal=False,
        )
        query_46 = key_45 = value_41 = item_39 = None
        transpose_23 = attn_output_56.transpose(1, 2)
        attn_output_56 = None
        attn_output_57 = transpose_23.contiguous()
        transpose_23 = None
        attn_output_58 = attn_output_57.reshape(1, 1, 4096, 128)
        attn_output_57 = None
        attn_output_59 = torch._C._nn.linear(
            attn_output_58,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_out_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_out_proj_parameters_bias_,
        )
        attn_output_58 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_out_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_out_proj_parameters_bias_ = (None)
        keys_2 = keys_1 + attn_output_59
        keys_1 = attn_output_59 = None
        item_40 = (
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
            item_40,
        )
        keys_2 = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm4_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm4_parameters_bias_ = (item_40) = (
            None
        )
        query_47 = queries_10 + point_embeddings
        point_embeddings = None
        key_46 = keys_3 + image_positional_embeddings_3
        image_positional_embeddings_3 = None
        query_48 = torch._C._nn.linear(
            query_47,
            l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_q_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_q_proj_parameters_bias_,
        )
        query_47 = l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_q_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_q_proj_parameters_bias_ = (None)
        key_47 = torch._C._nn.linear(
            key_46,
            l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_k_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_k_proj_parameters_bias_,
        )
        key_46 = l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_k_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_k_proj_parameters_bias_ = (None)
        value_42 = torch._C._nn.linear(
            keys_3,
            l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_v_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_v_proj_parameters_bias_,
        )
        l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_v_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_v_proj_parameters_bias_ = (None)
        hidden_states_140 = query_48.reshape(1, 5, 8, 16)
        query_48 = None
        query_49 = hidden_states_140.transpose(1, 2)
        hidden_states_140 = None
        hidden_states_141 = key_47.reshape(1, 4096, 8, 16)
        key_47 = None
        key_48 = hidden_states_141.transpose(1, 2)
        hidden_states_141 = None
        hidden_states_142 = value_42.reshape(1, 4096, 8, 16)
        value_42 = None
        value_43 = hidden_states_142.transpose(1, 2)
        hidden_states_142 = None
        query_50 = query_49.contiguous()
        query_49 = None
        key_49 = key_48.contiguous()
        key_48 = None
        value_44 = value_43.contiguous()
        value_43 = None
        item_41 = (
            l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_scaling.item()
        )
        l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_scaling = (
            None
        )
        attn_output_60 = torch._C._nn.scaled_dot_product_attention(
            query_50,
            key_49,
            value_44,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_41,
            is_causal=False,
        )
        query_50 = key_49 = value_44 = item_41 = None
        transpose_27 = attn_output_60.transpose(1, 2)
        attn_output_60 = None
        attn_output_61 = transpose_27.contiguous()
        transpose_27 = None
        attn_output_62 = attn_output_61.reshape(1, 1, 5, 128)
        attn_output_61 = None
        attn_output_63 = torch._C._nn.linear(
            attn_output_62,
            l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_out_proj_parameters_weight_,
            l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_out_proj_parameters_bias_,
        )
        attn_output_62 = l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_out_proj_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_out_proj_parameters_bias_ = (None)
        queries_11 = queries_10 + attn_output_63
        queries_10 = attn_output_63 = None
        item_42 = (
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
            item_42,
        )
        queries_11 = l_self_modules_mask_decoder_modules_transformer_modules_layer_norm_final_attn_parameters_weight_ = l_self_modules_mask_decoder_modules_transformer_modules_layer_norm_final_attn_parameters_bias_ = (item_42) = (
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
        sub_31 = x_8 - u_2
        pow_3 = sub_31.pow(2)
        sub_31 = None
        s_2 = pow_3.mean(1, keepdim=True)
        pow_3 = None
        sub_32 = x_8 - u_2
        x_8 = u_2 = None
        item_43 = l_self_modules_mask_decoder_modules_upscale_layer_norm_eps.item()
        l_self_modules_mask_decoder_modules_upscale_layer_norm_eps = None
        add_85 = s_2 + item_43
        s_2 = item_43 = None
        sqrt_2 = torch.sqrt(add_85)
        add_85 = None
        x_9 = sub_32 / sqrt_2
        sub_32 = sqrt_2 = None
        x_10 = x_9.to(dtype=torch.float32)
        x_9 = None
        getitem_154 = (
            l_self_modules_mask_decoder_modules_upscale_layer_norm_parameters_weight_[
                (slice(None, None, None), None, None)
            ]
        )
        l_self_modules_mask_decoder_modules_upscale_layer_norm_parameters_weight_ = None
        mul_53 = getitem_154 * x_10
        getitem_154 = x_10 = None
        getitem_155 = (
            l_self_modules_mask_decoder_modules_upscale_layer_norm_parameters_bias_[
                (slice(None, None, None), None, None)
            ]
        )
        l_self_modules_mask_decoder_modules_upscale_layer_norm_parameters_bias_ = None
        x_11 = mul_53 + getitem_155
        mul_53 = getitem_155 = None
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
        getitem_156 = mask_tokens_out[
            (
                slice(None, None, None),
                slice(None, None, None),
                0,
                slice(None, None, None),
            )
        ]
        hidden_states_143 = torch._C._nn.linear(
            getitem_156,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_proj_in_parameters_weight_,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_proj_in_parameters_bias_,
        )
        getitem_156 = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_proj_in_parameters_weight_ = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_proj_in_parameters_bias_ = (None)
        hidden_states_144 = torch.nn.functional.relu(hidden_states_143, inplace=False)
        hidden_states_143 = None
        linear_81 = torch._C._nn.linear(
            hidden_states_144,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_layers_modules_0_parameters_weight_,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_layers_modules_0_parameters_bias_,
        )
        hidden_states_144 = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_layers_modules_0_parameters_weight_ = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_layers_modules_0_parameters_bias_ = (None)
        hidden_states_145 = torch.nn.functional.relu(linear_81, inplace=False)
        linear_81 = None
        hidden_states_146 = torch._C._nn.linear(
            hidden_states_145,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_proj_out_parameters_weight_,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_proj_out_parameters_bias_,
        )
        hidden_states_145 = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_proj_out_parameters_weight_ = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_proj_out_parameters_bias_ = (None)
        getitem_157 = mask_tokens_out[
            (
                slice(None, None, None),
                slice(None, None, None),
                1,
                slice(None, None, None),
            )
        ]
        hidden_states_147 = torch._C._nn.linear(
            getitem_157,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_proj_in_parameters_weight_,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_proj_in_parameters_bias_,
        )
        getitem_157 = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_proj_in_parameters_weight_ = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_proj_in_parameters_bias_ = (None)
        hidden_states_148 = torch.nn.functional.relu(hidden_states_147, inplace=False)
        hidden_states_147 = None
        linear_84 = torch._C._nn.linear(
            hidden_states_148,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_layers_modules_0_parameters_weight_,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_layers_modules_0_parameters_bias_,
        )
        hidden_states_148 = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_layers_modules_0_parameters_weight_ = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_layers_modules_0_parameters_bias_ = (None)
        hidden_states_149 = torch.nn.functional.relu(linear_84, inplace=False)
        linear_84 = None
        hidden_states_150 = torch._C._nn.linear(
            hidden_states_149,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_proj_out_parameters_weight_,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_proj_out_parameters_bias_,
        )
        hidden_states_149 = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_proj_out_parameters_weight_ = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_proj_out_parameters_bias_ = (None)
        getitem_158 = mask_tokens_out[
            (
                slice(None, None, None),
                slice(None, None, None),
                2,
                slice(None, None, None),
            )
        ]
        hidden_states_151 = torch._C._nn.linear(
            getitem_158,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_proj_in_parameters_weight_,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_proj_in_parameters_bias_,
        )
        getitem_158 = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_proj_in_parameters_weight_ = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_proj_in_parameters_bias_ = (None)
        hidden_states_152 = torch.nn.functional.relu(hidden_states_151, inplace=False)
        hidden_states_151 = None
        linear_87 = torch._C._nn.linear(
            hidden_states_152,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_layers_modules_0_parameters_weight_,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_layers_modules_0_parameters_bias_,
        )
        hidden_states_152 = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_layers_modules_0_parameters_weight_ = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_layers_modules_0_parameters_bias_ = (None)
        hidden_states_153 = torch.nn.functional.relu(linear_87, inplace=False)
        linear_87 = None
        hidden_states_154 = torch._C._nn.linear(
            hidden_states_153,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_proj_out_parameters_weight_,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_proj_out_parameters_bias_,
        )
        hidden_states_153 = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_proj_out_parameters_weight_ = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_proj_out_parameters_bias_ = (None)
        getitem_159 = mask_tokens_out[
            (
                slice(None, None, None),
                slice(None, None, None),
                3,
                slice(None, None, None),
            )
        ]
        mask_tokens_out = None
        hidden_states_155 = torch._C._nn.linear(
            getitem_159,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_proj_in_parameters_weight_,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_proj_in_parameters_bias_,
        )
        getitem_159 = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_proj_in_parameters_weight_ = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_proj_in_parameters_bias_ = (None)
        hidden_states_156 = torch.nn.functional.relu(hidden_states_155, inplace=False)
        hidden_states_155 = None
        linear_90 = torch._C._nn.linear(
            hidden_states_156,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_layers_modules_0_parameters_weight_,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_layers_modules_0_parameters_bias_,
        )
        hidden_states_156 = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_layers_modules_0_parameters_weight_ = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_layers_modules_0_parameters_bias_ = (None)
        hidden_states_157 = torch.nn.functional.relu(linear_90, inplace=False)
        linear_90 = None
        hidden_states_158 = torch._C._nn.linear(
            hidden_states_157,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_proj_out_parameters_weight_,
            l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_proj_out_parameters_bias_,
        )
        hidden_states_157 = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_proj_out_parameters_weight_ = l_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_proj_out_parameters_bias_ = (None)
        hyper_in = torch.stack(
            [
                hidden_states_146,
                hidden_states_150,
                hidden_states_154,
                hidden_states_158,
            ],
            dim=2,
        )
        hidden_states_146 = (
            hidden_states_150
        ) = hidden_states_154 = hidden_states_158 = None
        upscaled_embedding_3 = upscaled_embedding_2.reshape(1, 1, 32, 65536)
        upscaled_embedding_2 = None
        matmul_1 = hyper_in @ upscaled_embedding_3
        hyper_in = upscaled_embedding_3 = None
        masks = matmul_1.reshape(1, 1, -1, 256, 256)
        matmul_1 = None
        hidden_states_159 = torch._C._nn.linear(
            iou_token_out,
            l_self_modules_mask_decoder_modules_iou_prediction_head_modules_proj_in_parameters_weight_,
            l_self_modules_mask_decoder_modules_iou_prediction_head_modules_proj_in_parameters_bias_,
        )
        iou_token_out = l_self_modules_mask_decoder_modules_iou_prediction_head_modules_proj_in_parameters_weight_ = l_self_modules_mask_decoder_modules_iou_prediction_head_modules_proj_in_parameters_bias_ = (None)
        hidden_states_160 = torch.nn.functional.relu(hidden_states_159, inplace=False)
        hidden_states_159 = None
        linear_93 = torch._C._nn.linear(
            hidden_states_160,
            l_self_modules_mask_decoder_modules_iou_prediction_head_modules_layers_modules_0_parameters_weight_,
            l_self_modules_mask_decoder_modules_iou_prediction_head_modules_layers_modules_0_parameters_bias_,
        )
        hidden_states_160 = l_self_modules_mask_decoder_modules_iou_prediction_head_modules_layers_modules_0_parameters_weight_ = l_self_modules_mask_decoder_modules_iou_prediction_head_modules_layers_modules_0_parameters_bias_ = (None)
        hidden_states_161 = torch.nn.functional.relu(linear_93, inplace=False)
        linear_93 = None
        hidden_states_162 = torch._C._nn.linear(
            hidden_states_161,
            l_self_modules_mask_decoder_modules_iou_prediction_head_modules_proj_out_parameters_weight_,
            l_self_modules_mask_decoder_modules_iou_prediction_head_modules_proj_out_parameters_bias_,
        )
        hidden_states_161 = l_self_modules_mask_decoder_modules_iou_prediction_head_modules_proj_out_parameters_weight_ = l_self_modules_mask_decoder_modules_iou_prediction_head_modules_proj_out_parameters_bias_ = (None)
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
        iou_pred = hidden_states_162[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        hidden_states_162 = None
        return (iou_pred, masks_1)
