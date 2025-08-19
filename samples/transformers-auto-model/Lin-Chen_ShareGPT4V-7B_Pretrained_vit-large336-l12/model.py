import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_kwargs_pixel_values_: torch.Tensor,
        L_self_modules_vision_model_modules_embeddings_modules_patch_embedding_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_embeddings_parameters_class_embedding_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_embeddings_buffers_position_ids_: torch.Tensor,
        L_self_modules_vision_model_modules_embeddings_modules_position_embedding_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_embeddings_modules_position_embedding_norm_type: torch.Tensor,
        L_self_modules_vision_model_modules_pre_layrnorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_pre_layrnorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_pre_layrnorm_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_post_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_post_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_post_layernorm_eps: torch.Tensor,
    ):
        l_kwargs_pixel_values_ = L_kwargs_pixel_values_
        l_self_modules_vision_model_modules_embeddings_modules_patch_embedding_parameters_weight_ = L_self_modules_vision_model_modules_embeddings_modules_patch_embedding_parameters_weight_
        l_self_modules_vision_model_modules_embeddings_parameters_class_embedding_ = (
            L_self_modules_vision_model_modules_embeddings_parameters_class_embedding_
        )
        l_self_modules_vision_model_modules_embeddings_buffers_position_ids_ = (
            L_self_modules_vision_model_modules_embeddings_buffers_position_ids_
        )
        l_self_modules_vision_model_modules_embeddings_modules_position_embedding_parameters_weight_ = L_self_modules_vision_model_modules_embeddings_modules_position_embedding_parameters_weight_
        l_self_modules_vision_model_modules_embeddings_modules_position_embedding_norm_type = L_self_modules_vision_model_modules_embeddings_modules_position_embedding_norm_type
        l_self_modules_vision_model_modules_pre_layrnorm_parameters_weight_ = (
            L_self_modules_vision_model_modules_pre_layrnorm_parameters_weight_
        )
        l_self_modules_vision_model_modules_pre_layrnorm_parameters_bias_ = (
            L_self_modules_vision_model_modules_pre_layrnorm_parameters_bias_
        )
        l_self_modules_vision_model_modules_pre_layrnorm_eps = (
            L_self_modules_vision_model_modules_pre_layrnorm_eps
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_post_layernorm_parameters_weight_ = (
            L_self_modules_vision_model_modules_post_layernorm_parameters_weight_
        )
        l_self_modules_vision_model_modules_post_layernorm_parameters_bias_ = (
            L_self_modules_vision_model_modules_post_layernorm_parameters_bias_
        )
        l_self_modules_vision_model_modules_post_layernorm_eps = (
            L_self_modules_vision_model_modules_post_layernorm_eps
        )
        to = l_kwargs_pixel_values_.to(dtype=torch.bfloat16)
        l_kwargs_pixel_values_ = None
        patch_embeds = torch.conv2d(
            to,
            l_self_modules_vision_model_modules_embeddings_modules_patch_embedding_parameters_weight_,
            None,
            (14, 14),
            (0, 0),
            (1, 1),
            1,
        )
        to = l_self_modules_vision_model_modules_embeddings_modules_patch_embedding_parameters_weight_ = (None)
        flatten = patch_embeds.flatten(2)
        patch_embeds = None
        patch_embeds_1 = flatten.transpose(1, 2)
        flatten = None
        class_embeds = l_self_modules_vision_model_modules_embeddings_parameters_class_embedding_.expand(
            1, 1, -1
        )
        l_self_modules_vision_model_modules_embeddings_parameters_class_embedding_ = (
            None
        )
        embeddings = torch.cat([class_embeds, patch_embeds_1], dim=1)
        class_embeds = patch_embeds_1 = None
        item = (
            l_self_modules_vision_model_modules_embeddings_modules_position_embedding_norm_type.item()
        )
        l_self_modules_vision_model_modules_embeddings_modules_position_embedding_norm_type = (
            None
        )
        embedding = torch.nn.functional.embedding(
            l_self_modules_vision_model_modules_embeddings_buffers_position_ids_,
            l_self_modules_vision_model_modules_embeddings_modules_position_embedding_parameters_weight_,
            None,
            None,
            item,
            False,
            False,
        )
        l_self_modules_vision_model_modules_embeddings_buffers_position_ids_ = l_self_modules_vision_model_modules_embeddings_modules_position_embedding_parameters_weight_ = (item) = (
            None
        )
        embeddings_1 = embeddings + embedding
        embeddings = embedding = None
        item_1 = l_self_modules_vision_model_modules_pre_layrnorm_eps.item()
        l_self_modules_vision_model_modules_pre_layrnorm_eps = None
        hidden_states = torch.nn.functional.layer_norm(
            embeddings_1,
            (1024,),
            l_self_modules_vision_model_modules_pre_layrnorm_parameters_weight_,
            l_self_modules_vision_model_modules_pre_layrnorm_parameters_bias_,
            item_1,
        )
        embeddings_1 = (
            l_self_modules_vision_model_modules_pre_layrnorm_parameters_weight_
        ) = (
            l_self_modules_vision_model_modules_pre_layrnorm_parameters_bias_
        ) = item_1 = None
        item_2 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_eps = (
            None
        )
        hidden_states_1 = torch.nn.functional.layer_norm(
            hidden_states,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_,
            item_2,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_ = (item_2) = (
            None
        )
        queries = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_1 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view = queries.view(1, 577, -1, 64)
        queries = None
        queries_1 = view.transpose(1, 2)
        view = None
        view_1 = keys.view(1, 577, -1, 64)
        keys = None
        keys_1 = view_1.transpose(1, 2)
        view_1 = None
        view_2 = values.view(1, 577, -1, 64)
        values = None
        values_1 = view_2.transpose(1, 2)
        view_2 = None
        query = queries_1.contiguous()
        queries_1 = None
        key = keys_1.contiguous()
        keys_1 = None
        value = values_1.contiguous()
        values_1 = None
        item_3 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_scale = (
            None
        )
        attn_output = torch._C._nn.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_3,
            is_causal=False,
        )
        query = key = value = item_3 = None
        transpose_4 = attn_output.transpose(1, 2)
        attn_output = None
        attn_output_1 = transpose_4.contiguous()
        transpose_4 = None
        reshape = attn_output_1.reshape(1, 577, 1024)
        attn_output_1 = None
        attn_output_2 = reshape.contiguous()
        reshape = None
        attn_output_3 = torch._C._nn.linear(
            attn_output_2,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_2 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_2 = hidden_states + attn_output_3
        hidden_states = attn_output_3 = None
        item_4 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_eps = (
            None
        )
        hidden_states_3 = torch.nn.functional.layer_norm(
            hidden_states_2,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_,
            item_4,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_ = (item_4) = (
            None
        )
        hidden_states_4 = torch._C._nn.linear(
            hidden_states_3,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_3 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul = 1.702 * hidden_states_4
        sigmoid = torch.sigmoid(mul)
        mul = None
        hidden_states_5 = hidden_states_4 * sigmoid
        hidden_states_4 = sigmoid = None
        hidden_states_6 = torch._C._nn.linear(
            hidden_states_5,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_5 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_7 = hidden_states_2 + hidden_states_6
        hidden_states_2 = hidden_states_6 = None
        item_5 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_eps = (
            None
        )
        hidden_states_8 = torch.nn.functional.layer_norm(
            hidden_states_7,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_bias_,
            item_5,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_bias_ = (item_5) = (
            None
        )
        queries_2 = torch._C._nn.linear(
            hidden_states_8,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_2 = torch._C._nn.linear(
            hidden_states_8,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_2 = torch._C._nn.linear(
            hidden_states_8,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_8 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_3 = queries_2.view(1, 577, -1, 64)
        queries_2 = None
        queries_3 = view_3.transpose(1, 2)
        view_3 = None
        view_4 = keys_2.view(1, 577, -1, 64)
        keys_2 = None
        keys_3 = view_4.transpose(1, 2)
        view_4 = None
        view_5 = values_2.view(1, 577, -1, 64)
        values_2 = None
        values_3 = view_5.transpose(1, 2)
        view_5 = None
        query_1 = queries_3.contiguous()
        queries_3 = None
        key_1 = keys_3.contiguous()
        keys_3 = None
        value_1 = values_3.contiguous()
        values_3 = None
        item_6 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_scale = (
            None
        )
        attn_output_4 = torch._C._nn.scaled_dot_product_attention(
            query_1,
            key_1,
            value_1,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_6,
            is_causal=False,
        )
        query_1 = key_1 = value_1 = item_6 = None
        transpose_8 = attn_output_4.transpose(1, 2)
        attn_output_4 = None
        attn_output_5 = transpose_8.contiguous()
        transpose_8 = None
        reshape_1 = attn_output_5.reshape(1, 577, 1024)
        attn_output_5 = None
        attn_output_6 = reshape_1.contiguous()
        reshape_1 = None
        attn_output_7 = torch._C._nn.linear(
            attn_output_6,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_6 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_9 = hidden_states_7 + attn_output_7
        hidden_states_7 = attn_output_7 = None
        item_7 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_eps = (
            None
        )
        hidden_states_10 = torch.nn.functional.layer_norm(
            hidden_states_9,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_bias_,
            item_7,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_bias_ = (item_7) = (
            None
        )
        hidden_states_11 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_10 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_2 = 1.702 * hidden_states_11
        sigmoid_1 = torch.sigmoid(mul_2)
        mul_2 = None
        hidden_states_12 = hidden_states_11 * sigmoid_1
        hidden_states_11 = sigmoid_1 = None
        hidden_states_13 = torch._C._nn.linear(
            hidden_states_12,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_12 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_14 = hidden_states_9 + hidden_states_13
        hidden_states_9 = hidden_states_13 = None
        item_8 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_eps = (
            None
        )
        hidden_states_15 = torch.nn.functional.layer_norm(
            hidden_states_14,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_bias_,
            item_8,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_bias_ = (item_8) = (
            None
        )
        queries_4 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_4 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_4 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_15 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_6 = queries_4.view(1, 577, -1, 64)
        queries_4 = None
        queries_5 = view_6.transpose(1, 2)
        view_6 = None
        view_7 = keys_4.view(1, 577, -1, 64)
        keys_4 = None
        keys_5 = view_7.transpose(1, 2)
        view_7 = None
        view_8 = values_4.view(1, 577, -1, 64)
        values_4 = None
        values_5 = view_8.transpose(1, 2)
        view_8 = None
        query_2 = queries_5.contiguous()
        queries_5 = None
        key_2 = keys_5.contiguous()
        keys_5 = None
        value_2 = values_5.contiguous()
        values_5 = None
        item_9 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_scale = (
            None
        )
        attn_output_8 = torch._C._nn.scaled_dot_product_attention(
            query_2,
            key_2,
            value_2,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_9,
            is_causal=False,
        )
        query_2 = key_2 = value_2 = item_9 = None
        transpose_12 = attn_output_8.transpose(1, 2)
        attn_output_8 = None
        attn_output_9 = transpose_12.contiguous()
        transpose_12 = None
        reshape_2 = attn_output_9.reshape(1, 577, 1024)
        attn_output_9 = None
        attn_output_10 = reshape_2.contiguous()
        reshape_2 = None
        attn_output_11 = torch._C._nn.linear(
            attn_output_10,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_10 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_16 = hidden_states_14 + attn_output_11
        hidden_states_14 = attn_output_11 = None
        item_10 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_eps = (
            None
        )
        hidden_states_17 = torch.nn.functional.layer_norm(
            hidden_states_16,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_bias_,
            item_10,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_bias_ = (item_10) = (
            None
        )
        hidden_states_18 = torch._C._nn.linear(
            hidden_states_17,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_17 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_4 = 1.702 * hidden_states_18
        sigmoid_2 = torch.sigmoid(mul_4)
        mul_4 = None
        hidden_states_19 = hidden_states_18 * sigmoid_2
        hidden_states_18 = sigmoid_2 = None
        hidden_states_20 = torch._C._nn.linear(
            hidden_states_19,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_19 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_21 = hidden_states_16 + hidden_states_20
        hidden_states_16 = hidden_states_20 = None
        item_11 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_eps = (
            None
        )
        hidden_states_22 = torch.nn.functional.layer_norm(
            hidden_states_21,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_bias_,
            item_11,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_bias_ = (item_11) = (
            None
        )
        queries_6 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_6 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_6 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_22 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_9 = queries_6.view(1, 577, -1, 64)
        queries_6 = None
        queries_7 = view_9.transpose(1, 2)
        view_9 = None
        view_10 = keys_6.view(1, 577, -1, 64)
        keys_6 = None
        keys_7 = view_10.transpose(1, 2)
        view_10 = None
        view_11 = values_6.view(1, 577, -1, 64)
        values_6 = None
        values_7 = view_11.transpose(1, 2)
        view_11 = None
        query_3 = queries_7.contiguous()
        queries_7 = None
        key_3 = keys_7.contiguous()
        keys_7 = None
        value_3 = values_7.contiguous()
        values_7 = None
        item_12 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_scale = (
            None
        )
        attn_output_12 = torch._C._nn.scaled_dot_product_attention(
            query_3,
            key_3,
            value_3,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_12,
            is_causal=False,
        )
        query_3 = key_3 = value_3 = item_12 = None
        transpose_16 = attn_output_12.transpose(1, 2)
        attn_output_12 = None
        attn_output_13 = transpose_16.contiguous()
        transpose_16 = None
        reshape_3 = attn_output_13.reshape(1, 577, 1024)
        attn_output_13 = None
        attn_output_14 = reshape_3.contiguous()
        reshape_3 = None
        attn_output_15 = torch._C._nn.linear(
            attn_output_14,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_14 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_23 = hidden_states_21 + attn_output_15
        hidden_states_21 = attn_output_15 = None
        item_13 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_eps = (
            None
        )
        hidden_states_24 = torch.nn.functional.layer_norm(
            hidden_states_23,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_bias_,
            item_13,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_bias_ = (item_13) = (
            None
        )
        hidden_states_25 = torch._C._nn.linear(
            hidden_states_24,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_24 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_6 = 1.702 * hidden_states_25
        sigmoid_3 = torch.sigmoid(mul_6)
        mul_6 = None
        hidden_states_26 = hidden_states_25 * sigmoid_3
        hidden_states_25 = sigmoid_3 = None
        hidden_states_27 = torch._C._nn.linear(
            hidden_states_26,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_26 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_28 = hidden_states_23 + hidden_states_27
        hidden_states_23 = hidden_states_27 = None
        item_14 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_eps = (
            None
        )
        hidden_states_29 = torch.nn.functional.layer_norm(
            hidden_states_28,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_bias_,
            item_14,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_bias_ = (item_14) = (
            None
        )
        queries_8 = torch._C._nn.linear(
            hidden_states_29,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_8 = torch._C._nn.linear(
            hidden_states_29,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_8 = torch._C._nn.linear(
            hidden_states_29,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_29 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_12 = queries_8.view(1, 577, -1, 64)
        queries_8 = None
        queries_9 = view_12.transpose(1, 2)
        view_12 = None
        view_13 = keys_8.view(1, 577, -1, 64)
        keys_8 = None
        keys_9 = view_13.transpose(1, 2)
        view_13 = None
        view_14 = values_8.view(1, 577, -1, 64)
        values_8 = None
        values_9 = view_14.transpose(1, 2)
        view_14 = None
        query_4 = queries_9.contiguous()
        queries_9 = None
        key_4 = keys_9.contiguous()
        keys_9 = None
        value_4 = values_9.contiguous()
        values_9 = None
        item_15 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_scale = (
            None
        )
        attn_output_16 = torch._C._nn.scaled_dot_product_attention(
            query_4,
            key_4,
            value_4,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_15,
            is_causal=False,
        )
        query_4 = key_4 = value_4 = item_15 = None
        transpose_20 = attn_output_16.transpose(1, 2)
        attn_output_16 = None
        attn_output_17 = transpose_20.contiguous()
        transpose_20 = None
        reshape_4 = attn_output_17.reshape(1, 577, 1024)
        attn_output_17 = None
        attn_output_18 = reshape_4.contiguous()
        reshape_4 = None
        attn_output_19 = torch._C._nn.linear(
            attn_output_18,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_18 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_30 = hidden_states_28 + attn_output_19
        hidden_states_28 = attn_output_19 = None
        item_16 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_eps = (
            None
        )
        hidden_states_31 = torch.nn.functional.layer_norm(
            hidden_states_30,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_bias_,
            item_16,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_bias_ = (item_16) = (
            None
        )
        hidden_states_32 = torch._C._nn.linear(
            hidden_states_31,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_31 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_8 = 1.702 * hidden_states_32
        sigmoid_4 = torch.sigmoid(mul_8)
        mul_8 = None
        hidden_states_33 = hidden_states_32 * sigmoid_4
        hidden_states_32 = sigmoid_4 = None
        hidden_states_34 = torch._C._nn.linear(
            hidden_states_33,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_33 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_35 = hidden_states_30 + hidden_states_34
        hidden_states_30 = hidden_states_34 = None
        item_17 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_eps = (
            None
        )
        hidden_states_36 = torch.nn.functional.layer_norm(
            hidden_states_35,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_bias_,
            item_17,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_bias_ = (item_17) = (
            None
        )
        queries_10 = torch._C._nn.linear(
            hidden_states_36,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_10 = torch._C._nn.linear(
            hidden_states_36,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_10 = torch._C._nn.linear(
            hidden_states_36,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_36 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_15 = queries_10.view(1, 577, -1, 64)
        queries_10 = None
        queries_11 = view_15.transpose(1, 2)
        view_15 = None
        view_16 = keys_10.view(1, 577, -1, 64)
        keys_10 = None
        keys_11 = view_16.transpose(1, 2)
        view_16 = None
        view_17 = values_10.view(1, 577, -1, 64)
        values_10 = None
        values_11 = view_17.transpose(1, 2)
        view_17 = None
        query_5 = queries_11.contiguous()
        queries_11 = None
        key_5 = keys_11.contiguous()
        keys_11 = None
        value_5 = values_11.contiguous()
        values_11 = None
        item_18 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_scale = (
            None
        )
        attn_output_20 = torch._C._nn.scaled_dot_product_attention(
            query_5,
            key_5,
            value_5,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_18,
            is_causal=False,
        )
        query_5 = key_5 = value_5 = item_18 = None
        transpose_24 = attn_output_20.transpose(1, 2)
        attn_output_20 = None
        attn_output_21 = transpose_24.contiguous()
        transpose_24 = None
        reshape_5 = attn_output_21.reshape(1, 577, 1024)
        attn_output_21 = None
        attn_output_22 = reshape_5.contiguous()
        reshape_5 = None
        attn_output_23 = torch._C._nn.linear(
            attn_output_22,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_22 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_37 = hidden_states_35 + attn_output_23
        hidden_states_35 = attn_output_23 = None
        item_19 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_eps = (
            None
        )
        hidden_states_38 = torch.nn.functional.layer_norm(
            hidden_states_37,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_bias_,
            item_19,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_bias_ = (item_19) = (
            None
        )
        hidden_states_39 = torch._C._nn.linear(
            hidden_states_38,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_38 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_10 = 1.702 * hidden_states_39
        sigmoid_5 = torch.sigmoid(mul_10)
        mul_10 = None
        hidden_states_40 = hidden_states_39 * sigmoid_5
        hidden_states_39 = sigmoid_5 = None
        hidden_states_41 = torch._C._nn.linear(
            hidden_states_40,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_40 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_42 = hidden_states_37 + hidden_states_41
        hidden_states_37 = hidden_states_41 = None
        item_20 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_eps = (
            None
        )
        hidden_states_43 = torch.nn.functional.layer_norm(
            hidden_states_42,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_bias_,
            item_20,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_bias_ = (item_20) = (
            None
        )
        queries_12 = torch._C._nn.linear(
            hidden_states_43,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_12 = torch._C._nn.linear(
            hidden_states_43,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_12 = torch._C._nn.linear(
            hidden_states_43,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_43 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_18 = queries_12.view(1, 577, -1, 64)
        queries_12 = None
        queries_13 = view_18.transpose(1, 2)
        view_18 = None
        view_19 = keys_12.view(1, 577, -1, 64)
        keys_12 = None
        keys_13 = view_19.transpose(1, 2)
        view_19 = None
        view_20 = values_12.view(1, 577, -1, 64)
        values_12 = None
        values_13 = view_20.transpose(1, 2)
        view_20 = None
        query_6 = queries_13.contiguous()
        queries_13 = None
        key_6 = keys_13.contiguous()
        keys_13 = None
        value_6 = values_13.contiguous()
        values_13 = None
        item_21 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_scale = (
            None
        )
        attn_output_24 = torch._C._nn.scaled_dot_product_attention(
            query_6,
            key_6,
            value_6,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_21,
            is_causal=False,
        )
        query_6 = key_6 = value_6 = item_21 = None
        transpose_28 = attn_output_24.transpose(1, 2)
        attn_output_24 = None
        attn_output_25 = transpose_28.contiguous()
        transpose_28 = None
        reshape_6 = attn_output_25.reshape(1, 577, 1024)
        attn_output_25 = None
        attn_output_26 = reshape_6.contiguous()
        reshape_6 = None
        attn_output_27 = torch._C._nn.linear(
            attn_output_26,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_26 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_44 = hidden_states_42 + attn_output_27
        hidden_states_42 = attn_output_27 = None
        item_22 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_eps = (
            None
        )
        hidden_states_45 = torch.nn.functional.layer_norm(
            hidden_states_44,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_bias_,
            item_22,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_bias_ = (item_22) = (
            None
        )
        hidden_states_46 = torch._C._nn.linear(
            hidden_states_45,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_45 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_12 = 1.702 * hidden_states_46
        sigmoid_6 = torch.sigmoid(mul_12)
        mul_12 = None
        hidden_states_47 = hidden_states_46 * sigmoid_6
        hidden_states_46 = sigmoid_6 = None
        hidden_states_48 = torch._C._nn.linear(
            hidden_states_47,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_47 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_49 = hidden_states_44 + hidden_states_48
        hidden_states_44 = hidden_states_48 = None
        item_23 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_eps = (
            None
        )
        hidden_states_50 = torch.nn.functional.layer_norm(
            hidden_states_49,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_bias_,
            item_23,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_bias_ = (item_23) = (
            None
        )
        queries_14 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_14 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_14 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_50 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_21 = queries_14.view(1, 577, -1, 64)
        queries_14 = None
        queries_15 = view_21.transpose(1, 2)
        view_21 = None
        view_22 = keys_14.view(1, 577, -1, 64)
        keys_14 = None
        keys_15 = view_22.transpose(1, 2)
        view_22 = None
        view_23 = values_14.view(1, 577, -1, 64)
        values_14 = None
        values_15 = view_23.transpose(1, 2)
        view_23 = None
        query_7 = queries_15.contiguous()
        queries_15 = None
        key_7 = keys_15.contiguous()
        keys_15 = None
        value_7 = values_15.contiguous()
        values_15 = None
        item_24 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_scale = (
            None
        )
        attn_output_28 = torch._C._nn.scaled_dot_product_attention(
            query_7,
            key_7,
            value_7,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_24,
            is_causal=False,
        )
        query_7 = key_7 = value_7 = item_24 = None
        transpose_32 = attn_output_28.transpose(1, 2)
        attn_output_28 = None
        attn_output_29 = transpose_32.contiguous()
        transpose_32 = None
        reshape_7 = attn_output_29.reshape(1, 577, 1024)
        attn_output_29 = None
        attn_output_30 = reshape_7.contiguous()
        reshape_7 = None
        attn_output_31 = torch._C._nn.linear(
            attn_output_30,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_30 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_51 = hidden_states_49 + attn_output_31
        hidden_states_49 = attn_output_31 = None
        item_25 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_eps = (
            None
        )
        hidden_states_52 = torch.nn.functional.layer_norm(
            hidden_states_51,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_bias_,
            item_25,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_bias_ = (item_25) = (
            None
        )
        hidden_states_53 = torch._C._nn.linear(
            hidden_states_52,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_52 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_14 = 1.702 * hidden_states_53
        sigmoid_7 = torch.sigmoid(mul_14)
        mul_14 = None
        hidden_states_54 = hidden_states_53 * sigmoid_7
        hidden_states_53 = sigmoid_7 = None
        hidden_states_55 = torch._C._nn.linear(
            hidden_states_54,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_54 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_56 = hidden_states_51 + hidden_states_55
        hidden_states_51 = hidden_states_55 = None
        item_26 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_eps = (
            None
        )
        hidden_states_57 = torch.nn.functional.layer_norm(
            hidden_states_56,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_bias_,
            item_26,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_bias_ = (item_26) = (
            None
        )
        queries_16 = torch._C._nn.linear(
            hidden_states_57,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_16 = torch._C._nn.linear(
            hidden_states_57,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_16 = torch._C._nn.linear(
            hidden_states_57,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_57 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_24 = queries_16.view(1, 577, -1, 64)
        queries_16 = None
        queries_17 = view_24.transpose(1, 2)
        view_24 = None
        view_25 = keys_16.view(1, 577, -1, 64)
        keys_16 = None
        keys_17 = view_25.transpose(1, 2)
        view_25 = None
        view_26 = values_16.view(1, 577, -1, 64)
        values_16 = None
        values_17 = view_26.transpose(1, 2)
        view_26 = None
        query_8 = queries_17.contiguous()
        queries_17 = None
        key_8 = keys_17.contiguous()
        keys_17 = None
        value_8 = values_17.contiguous()
        values_17 = None
        item_27 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_scale = (
            None
        )
        attn_output_32 = torch._C._nn.scaled_dot_product_attention(
            query_8,
            key_8,
            value_8,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_27,
            is_causal=False,
        )
        query_8 = key_8 = value_8 = item_27 = None
        transpose_36 = attn_output_32.transpose(1, 2)
        attn_output_32 = None
        attn_output_33 = transpose_36.contiguous()
        transpose_36 = None
        reshape_8 = attn_output_33.reshape(1, 577, 1024)
        attn_output_33 = None
        attn_output_34 = reshape_8.contiguous()
        reshape_8 = None
        attn_output_35 = torch._C._nn.linear(
            attn_output_34,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_34 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_58 = hidden_states_56 + attn_output_35
        hidden_states_56 = attn_output_35 = None
        item_28 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_eps = (
            None
        )
        hidden_states_59 = torch.nn.functional.layer_norm(
            hidden_states_58,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_bias_,
            item_28,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_bias_ = (item_28) = (
            None
        )
        hidden_states_60 = torch._C._nn.linear(
            hidden_states_59,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_59 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_16 = 1.702 * hidden_states_60
        sigmoid_8 = torch.sigmoid(mul_16)
        mul_16 = None
        hidden_states_61 = hidden_states_60 * sigmoid_8
        hidden_states_60 = sigmoid_8 = None
        hidden_states_62 = torch._C._nn.linear(
            hidden_states_61,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_61 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_63 = hidden_states_58 + hidden_states_62
        hidden_states_58 = hidden_states_62 = None
        item_29 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_eps = (
            None
        )
        hidden_states_64 = torch.nn.functional.layer_norm(
            hidden_states_63,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_bias_,
            item_29,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_bias_ = (item_29) = (
            None
        )
        queries_18 = torch._C._nn.linear(
            hidden_states_64,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_18 = torch._C._nn.linear(
            hidden_states_64,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_18 = torch._C._nn.linear(
            hidden_states_64,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_64 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_27 = queries_18.view(1, 577, -1, 64)
        queries_18 = None
        queries_19 = view_27.transpose(1, 2)
        view_27 = None
        view_28 = keys_18.view(1, 577, -1, 64)
        keys_18 = None
        keys_19 = view_28.transpose(1, 2)
        view_28 = None
        view_29 = values_18.view(1, 577, -1, 64)
        values_18 = None
        values_19 = view_29.transpose(1, 2)
        view_29 = None
        query_9 = queries_19.contiguous()
        queries_19 = None
        key_9 = keys_19.contiguous()
        keys_19 = None
        value_9 = values_19.contiguous()
        values_19 = None
        item_30 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_scale = (
            None
        )
        attn_output_36 = torch._C._nn.scaled_dot_product_attention(
            query_9,
            key_9,
            value_9,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_30,
            is_causal=False,
        )
        query_9 = key_9 = value_9 = item_30 = None
        transpose_40 = attn_output_36.transpose(1, 2)
        attn_output_36 = None
        attn_output_37 = transpose_40.contiguous()
        transpose_40 = None
        reshape_9 = attn_output_37.reshape(1, 577, 1024)
        attn_output_37 = None
        attn_output_38 = reshape_9.contiguous()
        reshape_9 = None
        attn_output_39 = torch._C._nn.linear(
            attn_output_38,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_38 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_65 = hidden_states_63 + attn_output_39
        hidden_states_63 = attn_output_39 = None
        item_31 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_eps = (
            None
        )
        hidden_states_66 = torch.nn.functional.layer_norm(
            hidden_states_65,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_bias_,
            item_31,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_bias_ = (item_31) = (
            None
        )
        hidden_states_67 = torch._C._nn.linear(
            hidden_states_66,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_66 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_18 = 1.702 * hidden_states_67
        sigmoid_9 = torch.sigmoid(mul_18)
        mul_18 = None
        hidden_states_68 = hidden_states_67 * sigmoid_9
        hidden_states_67 = sigmoid_9 = None
        hidden_states_69 = torch._C._nn.linear(
            hidden_states_68,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_68 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_70 = hidden_states_65 + hidden_states_69
        hidden_states_65 = hidden_states_69 = None
        item_32 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_eps = (
            None
        )
        hidden_states_71 = torch.nn.functional.layer_norm(
            hidden_states_70,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_bias_,
            item_32,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_bias_ = (item_32) = (
            None
        )
        queries_20 = torch._C._nn.linear(
            hidden_states_71,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_20 = torch._C._nn.linear(
            hidden_states_71,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_20 = torch._C._nn.linear(
            hidden_states_71,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_71 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_30 = queries_20.view(1, 577, -1, 64)
        queries_20 = None
        queries_21 = view_30.transpose(1, 2)
        view_30 = None
        view_31 = keys_20.view(1, 577, -1, 64)
        keys_20 = None
        keys_21 = view_31.transpose(1, 2)
        view_31 = None
        view_32 = values_20.view(1, 577, -1, 64)
        values_20 = None
        values_21 = view_32.transpose(1, 2)
        view_32 = None
        query_10 = queries_21.contiguous()
        queries_21 = None
        key_10 = keys_21.contiguous()
        keys_21 = None
        value_10 = values_21.contiguous()
        values_21 = None
        item_33 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_scale = (
            None
        )
        attn_output_40 = torch._C._nn.scaled_dot_product_attention(
            query_10,
            key_10,
            value_10,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_33,
            is_causal=False,
        )
        query_10 = key_10 = value_10 = item_33 = None
        transpose_44 = attn_output_40.transpose(1, 2)
        attn_output_40 = None
        attn_output_41 = transpose_44.contiguous()
        transpose_44 = None
        reshape_10 = attn_output_41.reshape(1, 577, 1024)
        attn_output_41 = None
        attn_output_42 = reshape_10.contiguous()
        reshape_10 = None
        attn_output_43 = torch._C._nn.linear(
            attn_output_42,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_42 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_72 = hidden_states_70 + attn_output_43
        hidden_states_70 = attn_output_43 = None
        item_34 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_eps = (
            None
        )
        hidden_states_73 = torch.nn.functional.layer_norm(
            hidden_states_72,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_bias_,
            item_34,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_bias_ = (item_34) = (
            None
        )
        hidden_states_74 = torch._C._nn.linear(
            hidden_states_73,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_73 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_20 = 1.702 * hidden_states_74
        sigmoid_10 = torch.sigmoid(mul_20)
        mul_20 = None
        hidden_states_75 = hidden_states_74 * sigmoid_10
        hidden_states_74 = sigmoid_10 = None
        hidden_states_76 = torch._C._nn.linear(
            hidden_states_75,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_75 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_77 = hidden_states_72 + hidden_states_76
        hidden_states_72 = hidden_states_76 = None
        item_35 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_eps = (
            None
        )
        hidden_states_78 = torch.nn.functional.layer_norm(
            hidden_states_77,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_bias_,
            item_35,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_bias_ = (item_35) = (
            None
        )
        queries_22 = torch._C._nn.linear(
            hidden_states_78,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_22 = torch._C._nn.linear(
            hidden_states_78,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_22 = torch._C._nn.linear(
            hidden_states_78,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_78 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_33 = queries_22.view(1, 577, -1, 64)
        queries_22 = None
        queries_23 = view_33.transpose(1, 2)
        view_33 = None
        view_34 = keys_22.view(1, 577, -1, 64)
        keys_22 = None
        keys_23 = view_34.transpose(1, 2)
        view_34 = None
        view_35 = values_22.view(1, 577, -1, 64)
        values_22 = None
        values_23 = view_35.transpose(1, 2)
        view_35 = None
        query_11 = queries_23.contiguous()
        queries_23 = None
        key_11 = keys_23.contiguous()
        keys_23 = None
        value_11 = values_23.contiguous()
        values_23 = None
        item_36 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_scale = (
            None
        )
        attn_output_44 = torch._C._nn.scaled_dot_product_attention(
            query_11,
            key_11,
            value_11,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_36,
            is_causal=False,
        )
        query_11 = key_11 = value_11 = item_36 = None
        transpose_48 = attn_output_44.transpose(1, 2)
        attn_output_44 = None
        attn_output_45 = transpose_48.contiguous()
        transpose_48 = None
        reshape_11 = attn_output_45.reshape(1, 577, 1024)
        attn_output_45 = None
        attn_output_46 = reshape_11.contiguous()
        reshape_11 = None
        attn_output_47 = torch._C._nn.linear(
            attn_output_46,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_46 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_79 = hidden_states_77 + attn_output_47
        hidden_states_77 = attn_output_47 = None
        item_37 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_eps = (
            None
        )
        hidden_states_80 = torch.nn.functional.layer_norm(
            hidden_states_79,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_bias_,
            item_37,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_bias_ = (item_37) = (
            None
        )
        hidden_states_81 = torch._C._nn.linear(
            hidden_states_80,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_80 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_22 = 1.702 * hidden_states_81
        sigmoid_11 = torch.sigmoid(mul_22)
        mul_22 = None
        hidden_states_82 = hidden_states_81 * sigmoid_11
        hidden_states_81 = sigmoid_11 = None
        hidden_states_83 = torch._C._nn.linear(
            hidden_states_82,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_82 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_84 = hidden_states_79 + hidden_states_83
        hidden_states_79 = hidden_states_83 = None
        item_38 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_eps = (
            None
        )
        hidden_states_85 = torch.nn.functional.layer_norm(
            hidden_states_84,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_bias_,
            item_38,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_bias_ = (item_38) = (
            None
        )
        queries_24 = torch._C._nn.linear(
            hidden_states_85,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_24 = torch._C._nn.linear(
            hidden_states_85,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_24 = torch._C._nn.linear(
            hidden_states_85,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_85 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_36 = queries_24.view(1, 577, -1, 64)
        queries_24 = None
        queries_25 = view_36.transpose(1, 2)
        view_36 = None
        view_37 = keys_24.view(1, 577, -1, 64)
        keys_24 = None
        keys_25 = view_37.transpose(1, 2)
        view_37 = None
        view_38 = values_24.view(1, 577, -1, 64)
        values_24 = None
        values_25 = view_38.transpose(1, 2)
        view_38 = None
        query_12 = queries_25.contiguous()
        queries_25 = None
        key_12 = keys_25.contiguous()
        keys_25 = None
        value_12 = values_25.contiguous()
        values_25 = None
        item_39 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_scale = (
            None
        )
        attn_output_48 = torch._C._nn.scaled_dot_product_attention(
            query_12,
            key_12,
            value_12,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_39,
            is_causal=False,
        )
        query_12 = key_12 = value_12 = item_39 = None
        transpose_52 = attn_output_48.transpose(1, 2)
        attn_output_48 = None
        attn_output_49 = transpose_52.contiguous()
        transpose_52 = None
        reshape_12 = attn_output_49.reshape(1, 577, 1024)
        attn_output_49 = None
        attn_output_50 = reshape_12.contiguous()
        reshape_12 = None
        attn_output_51 = torch._C._nn.linear(
            attn_output_50,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_50 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_86 = hidden_states_84 + attn_output_51
        hidden_states_84 = attn_output_51 = None
        item_40 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_eps = (
            None
        )
        hidden_states_87 = torch.nn.functional.layer_norm(
            hidden_states_86,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_bias_,
            item_40,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_bias_ = (item_40) = (
            None
        )
        hidden_states_88 = torch._C._nn.linear(
            hidden_states_87,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_87 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_24 = 1.702 * hidden_states_88
        sigmoid_12 = torch.sigmoid(mul_24)
        mul_24 = None
        hidden_states_89 = hidden_states_88 * sigmoid_12
        hidden_states_88 = sigmoid_12 = None
        hidden_states_90 = torch._C._nn.linear(
            hidden_states_89,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_89 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_91 = hidden_states_86 + hidden_states_90
        hidden_states_86 = hidden_states_90 = None
        item_41 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_eps = (
            None
        )
        hidden_states_92 = torch.nn.functional.layer_norm(
            hidden_states_91,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_bias_,
            item_41,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_bias_ = (item_41) = (
            None
        )
        queries_26 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_26 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_26 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_92 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_39 = queries_26.view(1, 577, -1, 64)
        queries_26 = None
        queries_27 = view_39.transpose(1, 2)
        view_39 = None
        view_40 = keys_26.view(1, 577, -1, 64)
        keys_26 = None
        keys_27 = view_40.transpose(1, 2)
        view_40 = None
        view_41 = values_26.view(1, 577, -1, 64)
        values_26 = None
        values_27 = view_41.transpose(1, 2)
        view_41 = None
        query_13 = queries_27.contiguous()
        queries_27 = None
        key_13 = keys_27.contiguous()
        keys_27 = None
        value_13 = values_27.contiguous()
        values_27 = None
        item_42 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_scale = (
            None
        )
        attn_output_52 = torch._C._nn.scaled_dot_product_attention(
            query_13,
            key_13,
            value_13,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_42,
            is_causal=False,
        )
        query_13 = key_13 = value_13 = item_42 = None
        transpose_56 = attn_output_52.transpose(1, 2)
        attn_output_52 = None
        attn_output_53 = transpose_56.contiguous()
        transpose_56 = None
        reshape_13 = attn_output_53.reshape(1, 577, 1024)
        attn_output_53 = None
        attn_output_54 = reshape_13.contiguous()
        reshape_13 = None
        attn_output_55 = torch._C._nn.linear(
            attn_output_54,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_54 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_93 = hidden_states_91 + attn_output_55
        hidden_states_91 = attn_output_55 = None
        item_43 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_eps = (
            None
        )
        hidden_states_94 = torch.nn.functional.layer_norm(
            hidden_states_93,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_bias_,
            item_43,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_bias_ = (item_43) = (
            None
        )
        hidden_states_95 = torch._C._nn.linear(
            hidden_states_94,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_94 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_26 = 1.702 * hidden_states_95
        sigmoid_13 = torch.sigmoid(mul_26)
        mul_26 = None
        hidden_states_96 = hidden_states_95 * sigmoid_13
        hidden_states_95 = sigmoid_13 = None
        hidden_states_97 = torch._C._nn.linear(
            hidden_states_96,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_96 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_98 = hidden_states_93 + hidden_states_97
        hidden_states_93 = hidden_states_97 = None
        item_44 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_eps = (
            None
        )
        hidden_states_99 = torch.nn.functional.layer_norm(
            hidden_states_98,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_bias_,
            item_44,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_bias_ = (item_44) = (
            None
        )
        queries_28 = torch._C._nn.linear(
            hidden_states_99,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_28 = torch._C._nn.linear(
            hidden_states_99,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_28 = torch._C._nn.linear(
            hidden_states_99,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_99 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_42 = queries_28.view(1, 577, -1, 64)
        queries_28 = None
        queries_29 = view_42.transpose(1, 2)
        view_42 = None
        view_43 = keys_28.view(1, 577, -1, 64)
        keys_28 = None
        keys_29 = view_43.transpose(1, 2)
        view_43 = None
        view_44 = values_28.view(1, 577, -1, 64)
        values_28 = None
        values_29 = view_44.transpose(1, 2)
        view_44 = None
        query_14 = queries_29.contiguous()
        queries_29 = None
        key_14 = keys_29.contiguous()
        keys_29 = None
        value_14 = values_29.contiguous()
        values_29 = None
        item_45 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_scale = (
            None
        )
        attn_output_56 = torch._C._nn.scaled_dot_product_attention(
            query_14,
            key_14,
            value_14,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_45,
            is_causal=False,
        )
        query_14 = key_14 = value_14 = item_45 = None
        transpose_60 = attn_output_56.transpose(1, 2)
        attn_output_56 = None
        attn_output_57 = transpose_60.contiguous()
        transpose_60 = None
        reshape_14 = attn_output_57.reshape(1, 577, 1024)
        attn_output_57 = None
        attn_output_58 = reshape_14.contiguous()
        reshape_14 = None
        attn_output_59 = torch._C._nn.linear(
            attn_output_58,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_58 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_100 = hidden_states_98 + attn_output_59
        hidden_states_98 = attn_output_59 = None
        item_46 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_eps = (
            None
        )
        hidden_states_101 = torch.nn.functional.layer_norm(
            hidden_states_100,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_bias_,
            item_46,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_bias_ = (item_46) = (
            None
        )
        hidden_states_102 = torch._C._nn.linear(
            hidden_states_101,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_101 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_28 = 1.702 * hidden_states_102
        sigmoid_14 = torch.sigmoid(mul_28)
        mul_28 = None
        hidden_states_103 = hidden_states_102 * sigmoid_14
        hidden_states_102 = sigmoid_14 = None
        hidden_states_104 = torch._C._nn.linear(
            hidden_states_103,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_103 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_105 = hidden_states_100 + hidden_states_104
        hidden_states_100 = hidden_states_104 = None
        item_47 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_eps = (
            None
        )
        hidden_states_106 = torch.nn.functional.layer_norm(
            hidden_states_105,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_bias_,
            item_47,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_bias_ = (item_47) = (
            None
        )
        queries_30 = torch._C._nn.linear(
            hidden_states_106,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_30 = torch._C._nn.linear(
            hidden_states_106,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_30 = torch._C._nn.linear(
            hidden_states_106,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_106 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_45 = queries_30.view(1, 577, -1, 64)
        queries_30 = None
        queries_31 = view_45.transpose(1, 2)
        view_45 = None
        view_46 = keys_30.view(1, 577, -1, 64)
        keys_30 = None
        keys_31 = view_46.transpose(1, 2)
        view_46 = None
        view_47 = values_30.view(1, 577, -1, 64)
        values_30 = None
        values_31 = view_47.transpose(1, 2)
        view_47 = None
        query_15 = queries_31.contiguous()
        queries_31 = None
        key_15 = keys_31.contiguous()
        keys_31 = None
        value_15 = values_31.contiguous()
        values_31 = None
        item_48 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_scale = (
            None
        )
        attn_output_60 = torch._C._nn.scaled_dot_product_attention(
            query_15,
            key_15,
            value_15,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_48,
            is_causal=False,
        )
        query_15 = key_15 = value_15 = item_48 = None
        transpose_64 = attn_output_60.transpose(1, 2)
        attn_output_60 = None
        attn_output_61 = transpose_64.contiguous()
        transpose_64 = None
        reshape_15 = attn_output_61.reshape(1, 577, 1024)
        attn_output_61 = None
        attn_output_62 = reshape_15.contiguous()
        reshape_15 = None
        attn_output_63 = torch._C._nn.linear(
            attn_output_62,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_62 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_107 = hidden_states_105 + attn_output_63
        hidden_states_105 = attn_output_63 = None
        item_49 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_eps = (
            None
        )
        hidden_states_108 = torch.nn.functional.layer_norm(
            hidden_states_107,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_bias_,
            item_49,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_bias_ = (item_49) = (
            None
        )
        hidden_states_109 = torch._C._nn.linear(
            hidden_states_108,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_108 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_30 = 1.702 * hidden_states_109
        sigmoid_15 = torch.sigmoid(mul_30)
        mul_30 = None
        hidden_states_110 = hidden_states_109 * sigmoid_15
        hidden_states_109 = sigmoid_15 = None
        hidden_states_111 = torch._C._nn.linear(
            hidden_states_110,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_110 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_112 = hidden_states_107 + hidden_states_111
        hidden_states_107 = hidden_states_111 = None
        item_50 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_eps = (
            None
        )
        hidden_states_113 = torch.nn.functional.layer_norm(
            hidden_states_112,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_bias_,
            item_50,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_bias_ = (item_50) = (
            None
        )
        queries_32 = torch._C._nn.linear(
            hidden_states_113,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_32 = torch._C._nn.linear(
            hidden_states_113,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_32 = torch._C._nn.linear(
            hidden_states_113,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_113 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_48 = queries_32.view(1, 577, -1, 64)
        queries_32 = None
        queries_33 = view_48.transpose(1, 2)
        view_48 = None
        view_49 = keys_32.view(1, 577, -1, 64)
        keys_32 = None
        keys_33 = view_49.transpose(1, 2)
        view_49 = None
        view_50 = values_32.view(1, 577, -1, 64)
        values_32 = None
        values_33 = view_50.transpose(1, 2)
        view_50 = None
        query_16 = queries_33.contiguous()
        queries_33 = None
        key_16 = keys_33.contiguous()
        keys_33 = None
        value_16 = values_33.contiguous()
        values_33 = None
        item_51 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_scale = (
            None
        )
        attn_output_64 = torch._C._nn.scaled_dot_product_attention(
            query_16,
            key_16,
            value_16,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_51,
            is_causal=False,
        )
        query_16 = key_16 = value_16 = item_51 = None
        transpose_68 = attn_output_64.transpose(1, 2)
        attn_output_64 = None
        attn_output_65 = transpose_68.contiguous()
        transpose_68 = None
        reshape_16 = attn_output_65.reshape(1, 577, 1024)
        attn_output_65 = None
        attn_output_66 = reshape_16.contiguous()
        reshape_16 = None
        attn_output_67 = torch._C._nn.linear(
            attn_output_66,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_66 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_114 = hidden_states_112 + attn_output_67
        hidden_states_112 = attn_output_67 = None
        item_52 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_eps = (
            None
        )
        hidden_states_115 = torch.nn.functional.layer_norm(
            hidden_states_114,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_bias_,
            item_52,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_bias_ = (item_52) = (
            None
        )
        hidden_states_116 = torch._C._nn.linear(
            hidden_states_115,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_115 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_32 = 1.702 * hidden_states_116
        sigmoid_16 = torch.sigmoid(mul_32)
        mul_32 = None
        hidden_states_117 = hidden_states_116 * sigmoid_16
        hidden_states_116 = sigmoid_16 = None
        hidden_states_118 = torch._C._nn.linear(
            hidden_states_117,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_117 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_119 = hidden_states_114 + hidden_states_118
        hidden_states_114 = hidden_states_118 = None
        item_53 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_eps = (
            None
        )
        hidden_states_120 = torch.nn.functional.layer_norm(
            hidden_states_119,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_bias_,
            item_53,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_bias_ = (item_53) = (
            None
        )
        queries_34 = torch._C._nn.linear(
            hidden_states_120,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_34 = torch._C._nn.linear(
            hidden_states_120,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_34 = torch._C._nn.linear(
            hidden_states_120,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_120 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_51 = queries_34.view(1, 577, -1, 64)
        queries_34 = None
        queries_35 = view_51.transpose(1, 2)
        view_51 = None
        view_52 = keys_34.view(1, 577, -1, 64)
        keys_34 = None
        keys_35 = view_52.transpose(1, 2)
        view_52 = None
        view_53 = values_34.view(1, 577, -1, 64)
        values_34 = None
        values_35 = view_53.transpose(1, 2)
        view_53 = None
        query_17 = queries_35.contiguous()
        queries_35 = None
        key_17 = keys_35.contiguous()
        keys_35 = None
        value_17 = values_35.contiguous()
        values_35 = None
        item_54 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_scale = (
            None
        )
        attn_output_68 = torch._C._nn.scaled_dot_product_attention(
            query_17,
            key_17,
            value_17,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_54,
            is_causal=False,
        )
        query_17 = key_17 = value_17 = item_54 = None
        transpose_72 = attn_output_68.transpose(1, 2)
        attn_output_68 = None
        attn_output_69 = transpose_72.contiguous()
        transpose_72 = None
        reshape_17 = attn_output_69.reshape(1, 577, 1024)
        attn_output_69 = None
        attn_output_70 = reshape_17.contiguous()
        reshape_17 = None
        attn_output_71 = torch._C._nn.linear(
            attn_output_70,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_70 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_121 = hidden_states_119 + attn_output_71
        hidden_states_119 = attn_output_71 = None
        item_55 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_eps = (
            None
        )
        hidden_states_122 = torch.nn.functional.layer_norm(
            hidden_states_121,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_bias_,
            item_55,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_bias_ = (item_55) = (
            None
        )
        hidden_states_123 = torch._C._nn.linear(
            hidden_states_122,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_122 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_34 = 1.702 * hidden_states_123
        sigmoid_17 = torch.sigmoid(mul_34)
        mul_34 = None
        hidden_states_124 = hidden_states_123 * sigmoid_17
        hidden_states_123 = sigmoid_17 = None
        hidden_states_125 = torch._C._nn.linear(
            hidden_states_124,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_124 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_126 = hidden_states_121 + hidden_states_125
        hidden_states_121 = hidden_states_125 = None
        item_56 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_eps = (
            None
        )
        hidden_states_127 = torch.nn.functional.layer_norm(
            hidden_states_126,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_bias_,
            item_56,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_bias_ = (item_56) = (
            None
        )
        queries_36 = torch._C._nn.linear(
            hidden_states_127,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_36 = torch._C._nn.linear(
            hidden_states_127,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_36 = torch._C._nn.linear(
            hidden_states_127,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_127 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_54 = queries_36.view(1, 577, -1, 64)
        queries_36 = None
        queries_37 = view_54.transpose(1, 2)
        view_54 = None
        view_55 = keys_36.view(1, 577, -1, 64)
        keys_36 = None
        keys_37 = view_55.transpose(1, 2)
        view_55 = None
        view_56 = values_36.view(1, 577, -1, 64)
        values_36 = None
        values_37 = view_56.transpose(1, 2)
        view_56 = None
        query_18 = queries_37.contiguous()
        queries_37 = None
        key_18 = keys_37.contiguous()
        keys_37 = None
        value_18 = values_37.contiguous()
        values_37 = None
        item_57 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_scale = (
            None
        )
        attn_output_72 = torch._C._nn.scaled_dot_product_attention(
            query_18,
            key_18,
            value_18,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_57,
            is_causal=False,
        )
        query_18 = key_18 = value_18 = item_57 = None
        transpose_76 = attn_output_72.transpose(1, 2)
        attn_output_72 = None
        attn_output_73 = transpose_76.contiguous()
        transpose_76 = None
        reshape_18 = attn_output_73.reshape(1, 577, 1024)
        attn_output_73 = None
        attn_output_74 = reshape_18.contiguous()
        reshape_18 = None
        attn_output_75 = torch._C._nn.linear(
            attn_output_74,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_74 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_128 = hidden_states_126 + attn_output_75
        hidden_states_126 = attn_output_75 = None
        item_58 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_eps = (
            None
        )
        hidden_states_129 = torch.nn.functional.layer_norm(
            hidden_states_128,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_bias_,
            item_58,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_bias_ = (item_58) = (
            None
        )
        hidden_states_130 = torch._C._nn.linear(
            hidden_states_129,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_129 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_36 = 1.702 * hidden_states_130
        sigmoid_18 = torch.sigmoid(mul_36)
        mul_36 = None
        hidden_states_131 = hidden_states_130 * sigmoid_18
        hidden_states_130 = sigmoid_18 = None
        hidden_states_132 = torch._C._nn.linear(
            hidden_states_131,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_131 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_133 = hidden_states_128 + hidden_states_132
        hidden_states_128 = hidden_states_132 = None
        item_59 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_eps = (
            None
        )
        hidden_states_134 = torch.nn.functional.layer_norm(
            hidden_states_133,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_bias_,
            item_59,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_bias_ = (item_59) = (
            None
        )
        queries_38 = torch._C._nn.linear(
            hidden_states_134,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_38 = torch._C._nn.linear(
            hidden_states_134,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_38 = torch._C._nn.linear(
            hidden_states_134,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_134 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_57 = queries_38.view(1, 577, -1, 64)
        queries_38 = None
        queries_39 = view_57.transpose(1, 2)
        view_57 = None
        view_58 = keys_38.view(1, 577, -1, 64)
        keys_38 = None
        keys_39 = view_58.transpose(1, 2)
        view_58 = None
        view_59 = values_38.view(1, 577, -1, 64)
        values_38 = None
        values_39 = view_59.transpose(1, 2)
        view_59 = None
        query_19 = queries_39.contiguous()
        queries_39 = None
        key_19 = keys_39.contiguous()
        keys_39 = None
        value_19 = values_39.contiguous()
        values_39 = None
        item_60 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_scale = (
            None
        )
        attn_output_76 = torch._C._nn.scaled_dot_product_attention(
            query_19,
            key_19,
            value_19,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_60,
            is_causal=False,
        )
        query_19 = key_19 = value_19 = item_60 = None
        transpose_80 = attn_output_76.transpose(1, 2)
        attn_output_76 = None
        attn_output_77 = transpose_80.contiguous()
        transpose_80 = None
        reshape_19 = attn_output_77.reshape(1, 577, 1024)
        attn_output_77 = None
        attn_output_78 = reshape_19.contiguous()
        reshape_19 = None
        attn_output_79 = torch._C._nn.linear(
            attn_output_78,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_78 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_135 = hidden_states_133 + attn_output_79
        hidden_states_133 = attn_output_79 = None
        item_61 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_eps = (
            None
        )
        hidden_states_136 = torch.nn.functional.layer_norm(
            hidden_states_135,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_bias_,
            item_61,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_bias_ = (item_61) = (
            None
        )
        hidden_states_137 = torch._C._nn.linear(
            hidden_states_136,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_136 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_38 = 1.702 * hidden_states_137
        sigmoid_19 = torch.sigmoid(mul_38)
        mul_38 = None
        hidden_states_138 = hidden_states_137 * sigmoid_19
        hidden_states_137 = sigmoid_19 = None
        hidden_states_139 = torch._C._nn.linear(
            hidden_states_138,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_138 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_140 = hidden_states_135 + hidden_states_139
        hidden_states_135 = hidden_states_139 = None
        item_62 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_eps = (
            None
        )
        hidden_states_141 = torch.nn.functional.layer_norm(
            hidden_states_140,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_bias_,
            item_62,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_bias_ = (item_62) = (
            None
        )
        queries_40 = torch._C._nn.linear(
            hidden_states_141,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_40 = torch._C._nn.linear(
            hidden_states_141,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_40 = torch._C._nn.linear(
            hidden_states_141,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_141 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_60 = queries_40.view(1, 577, -1, 64)
        queries_40 = None
        queries_41 = view_60.transpose(1, 2)
        view_60 = None
        view_61 = keys_40.view(1, 577, -1, 64)
        keys_40 = None
        keys_41 = view_61.transpose(1, 2)
        view_61 = None
        view_62 = values_40.view(1, 577, -1, 64)
        values_40 = None
        values_41 = view_62.transpose(1, 2)
        view_62 = None
        query_20 = queries_41.contiguous()
        queries_41 = None
        key_20 = keys_41.contiguous()
        keys_41 = None
        value_20 = values_41.contiguous()
        values_41 = None
        item_63 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_scale = (
            None
        )
        attn_output_80 = torch._C._nn.scaled_dot_product_attention(
            query_20,
            key_20,
            value_20,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_63,
            is_causal=False,
        )
        query_20 = key_20 = value_20 = item_63 = None
        transpose_84 = attn_output_80.transpose(1, 2)
        attn_output_80 = None
        attn_output_81 = transpose_84.contiguous()
        transpose_84 = None
        reshape_20 = attn_output_81.reshape(1, 577, 1024)
        attn_output_81 = None
        attn_output_82 = reshape_20.contiguous()
        reshape_20 = None
        attn_output_83 = torch._C._nn.linear(
            attn_output_82,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_82 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_142 = hidden_states_140 + attn_output_83
        hidden_states_140 = attn_output_83 = None
        item_64 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_eps = (
            None
        )
        hidden_states_143 = torch.nn.functional.layer_norm(
            hidden_states_142,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_bias_,
            item_64,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_bias_ = (item_64) = (
            None
        )
        hidden_states_144 = torch._C._nn.linear(
            hidden_states_143,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_143 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_40 = 1.702 * hidden_states_144
        sigmoid_20 = torch.sigmoid(mul_40)
        mul_40 = None
        hidden_states_145 = hidden_states_144 * sigmoid_20
        hidden_states_144 = sigmoid_20 = None
        hidden_states_146 = torch._C._nn.linear(
            hidden_states_145,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_145 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_147 = hidden_states_142 + hidden_states_146
        hidden_states_142 = hidden_states_146 = None
        item_65 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_eps = (
            None
        )
        hidden_states_148 = torch.nn.functional.layer_norm(
            hidden_states_147,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_bias_,
            item_65,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_bias_ = (item_65) = (
            None
        )
        queries_42 = torch._C._nn.linear(
            hidden_states_148,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_42 = torch._C._nn.linear(
            hidden_states_148,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_42 = torch._C._nn.linear(
            hidden_states_148,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_148 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_63 = queries_42.view(1, 577, -1, 64)
        queries_42 = None
        queries_43 = view_63.transpose(1, 2)
        view_63 = None
        view_64 = keys_42.view(1, 577, -1, 64)
        keys_42 = None
        keys_43 = view_64.transpose(1, 2)
        view_64 = None
        view_65 = values_42.view(1, 577, -1, 64)
        values_42 = None
        values_43 = view_65.transpose(1, 2)
        view_65 = None
        query_21 = queries_43.contiguous()
        queries_43 = None
        key_21 = keys_43.contiguous()
        keys_43 = None
        value_21 = values_43.contiguous()
        values_43 = None
        item_66 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_scale = (
            None
        )
        attn_output_84 = torch._C._nn.scaled_dot_product_attention(
            query_21,
            key_21,
            value_21,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_66,
            is_causal=False,
        )
        query_21 = key_21 = value_21 = item_66 = None
        transpose_88 = attn_output_84.transpose(1, 2)
        attn_output_84 = None
        attn_output_85 = transpose_88.contiguous()
        transpose_88 = None
        reshape_21 = attn_output_85.reshape(1, 577, 1024)
        attn_output_85 = None
        attn_output_86 = reshape_21.contiguous()
        reshape_21 = None
        attn_output_87 = torch._C._nn.linear(
            attn_output_86,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_86 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_149 = hidden_states_147 + attn_output_87
        hidden_states_147 = attn_output_87 = None
        item_67 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_eps = (
            None
        )
        hidden_states_150 = torch.nn.functional.layer_norm(
            hidden_states_149,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_bias_,
            item_67,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_bias_ = (item_67) = (
            None
        )
        hidden_states_151 = torch._C._nn.linear(
            hidden_states_150,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_150 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_42 = 1.702 * hidden_states_151
        sigmoid_21 = torch.sigmoid(mul_42)
        mul_42 = None
        hidden_states_152 = hidden_states_151 * sigmoid_21
        hidden_states_151 = sigmoid_21 = None
        hidden_states_153 = torch._C._nn.linear(
            hidden_states_152,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_152 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_154 = hidden_states_149 + hidden_states_153
        hidden_states_149 = hidden_states_153 = None
        item_68 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_eps = (
            None
        )
        hidden_states_155 = torch.nn.functional.layer_norm(
            hidden_states_154,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_bias_,
            item_68,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_bias_ = (item_68) = (
            None
        )
        queries_44 = torch._C._nn.linear(
            hidden_states_155,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_44 = torch._C._nn.linear(
            hidden_states_155,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_44 = torch._C._nn.linear(
            hidden_states_155,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_155 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_66 = queries_44.view(1, 577, -1, 64)
        queries_44 = None
        queries_45 = view_66.transpose(1, 2)
        view_66 = None
        view_67 = keys_44.view(1, 577, -1, 64)
        keys_44 = None
        keys_45 = view_67.transpose(1, 2)
        view_67 = None
        view_68 = values_44.view(1, 577, -1, 64)
        values_44 = None
        values_45 = view_68.transpose(1, 2)
        view_68 = None
        query_22 = queries_45.contiguous()
        queries_45 = None
        key_22 = keys_45.contiguous()
        keys_45 = None
        value_22 = values_45.contiguous()
        values_45 = None
        item_69 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_scale = (
            None
        )
        attn_output_88 = torch._C._nn.scaled_dot_product_attention(
            query_22,
            key_22,
            value_22,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_69,
            is_causal=False,
        )
        query_22 = key_22 = value_22 = item_69 = None
        transpose_92 = attn_output_88.transpose(1, 2)
        attn_output_88 = None
        attn_output_89 = transpose_92.contiguous()
        transpose_92 = None
        reshape_22 = attn_output_89.reshape(1, 577, 1024)
        attn_output_89 = None
        attn_output_90 = reshape_22.contiguous()
        reshape_22 = None
        attn_output_91 = torch._C._nn.linear(
            attn_output_90,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_90 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_156 = hidden_states_154 + attn_output_91
        hidden_states_154 = attn_output_91 = None
        item_70 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_eps = (
            None
        )
        hidden_states_157 = torch.nn.functional.layer_norm(
            hidden_states_156,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_bias_,
            item_70,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_bias_ = (item_70) = (
            None
        )
        hidden_states_158 = torch._C._nn.linear(
            hidden_states_157,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_157 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_44 = 1.702 * hidden_states_158
        sigmoid_22 = torch.sigmoid(mul_44)
        mul_44 = None
        hidden_states_159 = hidden_states_158 * sigmoid_22
        hidden_states_158 = sigmoid_22 = None
        hidden_states_160 = torch._C._nn.linear(
            hidden_states_159,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_159 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_161 = hidden_states_156 + hidden_states_160
        hidden_states_156 = hidden_states_160 = None
        item_71 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_eps = (
            None
        )
        hidden_states_162 = torch.nn.functional.layer_norm(
            hidden_states_161,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_bias_,
            item_71,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_bias_ = (item_71) = (
            None
        )
        queries_46 = torch._C._nn.linear(
            hidden_states_162,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_46 = torch._C._nn.linear(
            hidden_states_162,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_46 = torch._C._nn.linear(
            hidden_states_162,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_162 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_69 = queries_46.view(1, 577, -1, 64)
        queries_46 = None
        queries_47 = view_69.transpose(1, 2)
        view_69 = None
        view_70 = keys_46.view(1, 577, -1, 64)
        keys_46 = None
        keys_47 = view_70.transpose(1, 2)
        view_70 = None
        view_71 = values_46.view(1, 577, -1, 64)
        values_46 = None
        values_47 = view_71.transpose(1, 2)
        view_71 = None
        query_23 = queries_47.contiguous()
        queries_47 = None
        key_23 = keys_47.contiguous()
        keys_47 = None
        value_23 = values_47.contiguous()
        values_47 = None
        item_72 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_scale = (
            None
        )
        attn_output_92 = torch._C._nn.scaled_dot_product_attention(
            query_23,
            key_23,
            value_23,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_72,
            is_causal=False,
        )
        query_23 = key_23 = value_23 = item_72 = None
        transpose_96 = attn_output_92.transpose(1, 2)
        attn_output_92 = None
        attn_output_93 = transpose_96.contiguous()
        transpose_96 = None
        reshape_23 = attn_output_93.reshape(1, 577, 1024)
        attn_output_93 = None
        attn_output_94 = reshape_23.contiguous()
        reshape_23 = None
        attn_output_95 = torch._C._nn.linear(
            attn_output_94,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_94 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_163 = hidden_states_161 + attn_output_95
        hidden_states_161 = attn_output_95 = None
        item_73 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_eps = (
            None
        )
        hidden_states_164 = torch.nn.functional.layer_norm(
            hidden_states_163,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_bias_,
            item_73,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_bias_ = (item_73) = (
            None
        )
        hidden_states_165 = torch._C._nn.linear(
            hidden_states_164,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_164 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_46 = 1.702 * hidden_states_165
        sigmoid_23 = torch.sigmoid(mul_46)
        mul_46 = None
        hidden_states_166 = hidden_states_165 * sigmoid_23
        hidden_states_165 = sigmoid_23 = None
        hidden_states_167 = torch._C._nn.linear(
            hidden_states_166,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_166 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_168 = hidden_states_163 + hidden_states_167
        hidden_states_163 = hidden_states_167 = None
        pooled_output = hidden_states_168[
            (slice(None, None, None), 0, slice(None, None, None))
        ]
        item_74 = l_self_modules_vision_model_modules_post_layernorm_eps.item()
        l_self_modules_vision_model_modules_post_layernorm_eps = None
        pooled_output_1 = torch.nn.functional.layer_norm(
            pooled_output,
            (1024,),
            l_self_modules_vision_model_modules_post_layernorm_parameters_weight_,
            l_self_modules_vision_model_modules_post_layernorm_parameters_bias_,
            item_74,
        )
        pooled_output = (
            l_self_modules_vision_model_modules_post_layernorm_parameters_weight_
        ) = (
            l_self_modules_vision_model_modules_post_layernorm_parameters_bias_
        ) = item_74 = None
        return (hidden_states_168, pooled_output_1)
